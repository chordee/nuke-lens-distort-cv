// LensDistort.cpp
// Nuke NDK plugin: OpenCV-based lens distortion / undistortion
//
// Pattern: Iop + engine() + Tile — matches SimpleBlurCached NDK example.
// No NukeWrapper (that is only for PixelIop / pixel_engine pattern).
// Full frame is computed on first engine() call and cached per-channel.
//
// Knobs:
//   mode     : Undistort / Distort
//   is_fisheye : Enable OpenCV equidistant fisheye model
//   k1–k6    : Radial distortion coefficients (Brown-Conrady rational model)
//   p1, p2   : Tangential distortion (perspective) / fisheye k3, k4 (fisheye)
//   focal_x  : Focal length in pixels X  (0 = auto from image width)
//   focal_y  : Focal length in pixels Y  (0 = auto from image height)
//   center_x : Principal point X, normalized 0-1  (default 0.5)
//   center_y : Principal point Y, normalized 0-1  (default 0.5)
//   filter   : Nearest / Bilinear / Bicubic
//   alpha    : Fisheye balance parameter (0=crop, 1=keep all)
//
// Expand mode (two-JSON workflow):
//   When orig_w/orig_h > 0, the plugin uses two separate K matrices:
//   K_new (from primary JSON / focal_x/y + native_w/h) and
//   K_orig (from orig_* knobs, loaded via Load Original JSON button).
//   Undistort: input at K_orig size → output at K_new size
//   Distort  : input at K_new size → output at K_orig size

// On Windows, pin this DLL in memory before any static constructors run.
// Nuke may unload plugin DLLs that fail its validation checks, which would
// leave stale Op::Description pointers registered in DDImage. Pinning the
// module prevents FreeLibrary from unmapping our code.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
namespace {
    struct DllPinner {
        DllPinner() noexcept {
            HMODULE h = nullptr;
            // 'this' points to s_dll_pinner in the DLL's data segment —
            // a valid in-module address for FROM_ADDRESS to resolve the handle.
            // PIN permanently increments the refcount so FreeLibrary is a no-op.
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_PIN |
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                reinterpret_cast<LPCWSTR>(this), &h);
        }
    };
    // Run BEFORE user-level static constructors (and therefore before OpenCV's).
    #pragma init_seg(lib)
    static DllPinner s_dll_pinner;
}
#endif // _WIN32

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Tile.h"
#include "DDImage/Knobs.h"
#include "DDImage/Thread.h"
#include "DDImage/Format.h"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>   // getOptimalNewCameraMatrix, fisheye::* — no parallel_for_

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <map>

using namespace DD::Image;

// -----------------------------------------------------------------------------
static const char* const CLASS = "LensDistortCV";
static const char* const HELP  =
    "<p><b>LensDistortCV</b> — OpenCV lens distortion / undistortion.</p>"
    "<p>Applies or removes radial + tangential lens distortion using the "
    "Brown-Conrady model (same convention as OpenCV / ShotCalibrate).</p>"
    "<p><b>Undistort</b> — removes distortion (straightens footage).<br>"
    "<b>Distort</b>   — adds distortion (match a real lens).</p>"
    "<p>Coefficients follow OpenCV convention:<br>"
    "k1–k6 = radial &nbsp;&nbsp; p1, p2 = tangential (perspective) / fisheye k3, k4 (fisheye)</p>"
    "<p><b>Fisheye model</b> — equidistant OpenCV fisheye.<br>"
    "k1/k2 map to fisheye k1/k2; p1/p2 map to fisheye k3/k4 (no tangential).<br>"
    "Alpha controls the balance parameter for estimateNewCameraMatrixForUndistortRectify.</p>"
    "<p><b>Expand mode</b> — load two JSON files (original + undistorted).<br>"
    "When orig_w/orig_h &gt; 0, the plugin maps between two different canvas sizes.</p>";

static const char* const MODE_NAMES[]   = { "Undistort", "Distort",  nullptr };
static const char* const FILTER_NAMES[] = { "Nearest",   "Bilinear", "Bicubic", nullptr };

// -----------------------------------------------------------------------------
class LensDistort : public Iop
{
    int    _mode;
    bool   _isFisheye;
    double _k1, _k2, _k3;
    double _k4, _k5, _k6;
    double _p1, _p2;
    double _focalX, _focalY;
    double _centerX, _centerY;
    int    _filter;

    // fisheye: balance parameter for estimateNewCameraMatrixForUndistortRectify
    // perspective: unused (new_K = K directly, matching Python K.copy() behaviour)
    double _alpha;

    // NeRFStudio JSON import
    const char* _jsonFile;

    // Native (calibration) resolution for K_new — used to scale focal lengths.
    // 0 = disabled (treat focal_x/focal_y as absolute pixels at current res).
    int _nativeW, _nativeH;

    // Expand-mode: K_orig parameters (original distorted camera).
    // When _origW > 0 && _origH > 0, expand mode is active:
    //   Undistort: input is _origW×_origH → output is _nativeW×_nativeH
    //   Distort  : input is _nativeW×_nativeH → output is _origW×_origH
    double _origFocalX, _origFocalY;
    double _origCenterX, _origCenterY;
    int    _origW, _origH;
    const char* _origJsonFile;

    // Computed optimal new camera matrix (filled in _validate, displayed read-only)
    double _newFx, _newFy, _newCx, _newCy;

    // Persistent output format for expand mode.
    // Must outlive _validate() because info_.format() stores a pointer to it.
    Format _outputFmt;

    // Full-frame output cache, one cv::Mat per channel
    Lock                       _lock;
    bool                       _isFirstTime;
    std::map<Channel, cv::Mat> _outputCache;   // top-down float32

public:
    explicit LensDistort(Node* node)
        : Iop(node)
        , _mode(0)
        , _isFisheye(false)
        , _k1(0), _k2(0), _k3(0)
        , _k4(0), _k5(0), _k6(0)
        , _p1(0), _p2(0)
        , _focalX(0), _focalY(0)
        , _centerX(0.5), _centerY(0.5)
        , _filter(1)
        , _alpha(1.0)
        , _jsonFile("")
        , _nativeW(0), _nativeH(0)
        , _origFocalX(0), _origFocalY(0)
        , _origCenterX(0.5), _origCenterY(0.5)
        , _origW(0), _origH(0)
        , _origJsonFile("")
        , _newFx(0), _newFy(0), _newCx(0), _newCy(0)
        , _isFirstTime(true)
    {}

    // ── Knobs ─────────────────────────────────────────────────────────────────
    void knobs(Knob_Callback f) override
    {
        Enumeration_knob(f, &_mode, MODE_NAMES, "mode", "Mode");
        Tooltip(f, "Undistort removes distortion. Distort adds it.");

        Bool_knob(f, &_isFisheye, "is_fisheye", "Fisheye Model");
        Tooltip(f, "Enable OpenCV equidistant fisheye lens model.\n"
                   "When enabled:\n"
                   "  k1/k2  \xe2\x86\x92 fisheye k1/k2\n"
                   "  p1/p2  \xe2\x86\x92 fisheye k3/k4  (tangential terms unused)\n"
                   "  k3\xe2\x80\x93k6  \xe2\x86\x92 ignored\n"
                   "  Alpha  \xe2\x86\x92 balance for estimateNewCameraMatrixForUndistortRectify\n"
                   "When disabled: standard Brown-Conrady perspective model.\n"
                   "Note: expand mode (orig_w/h) is not supported in fisheye mode.");

        Divider(f, "Distortion Coefficients");

        Double_knob(f, &_k1, "k1", "k1");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Radial distortion k1.");
        Double_knob(f, &_k2, "k2", "k2");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Radial distortion k2.");
        Double_knob(f, &_k3, "k3", "k3");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Radial distortion k3 (numerator, r^6 term). Perspective mode only.");
        Double_knob(f, &_k4, "k4", "k4");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Rational denominator k4 (r^2). 0 = disabled. Perspective mode only.");
        Double_knob(f, &_k5, "k5", "k5");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Rational denominator k5 (r^4). 0 = disabled. Perspective mode only.");
        Double_knob(f, &_k6, "k6", "k6");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Rational denominator k6 (r^6). 0 = disabled. Perspective mode only.");
        Double_knob(f, &_p1, "p1", "p1");
        SetRange(f, -0.5, 0.5);
        Tooltip(f, "Tangential distortion p1 (perspective mode).\n"
                   "Fisheye mode: acts as fisheye k3 coefficient.");
        Double_knob(f, &_p2, "p2", "p2");
        SetRange(f, -0.5, 0.5);
        Tooltip(f, "Tangential distortion p2 (perspective mode).\n"
                   "Fisheye mode: acts as fisheye k4 coefficient.");

        Divider(f, "Camera Intrinsics");

        Double_knob(f, &_focalX, "focal_x", "Focal X");
        SetRange(f, 0, 10000);
        Tooltip(f, "Focal length in pixels X. 0 = auto (image width).\n"
                   "In expand mode: this is K_new focal X (from undistorted JSON).");
        Double_knob(f, &_focalY, "focal_y", "Focal Y");
        SetRange(f, 0, 10000);
        Tooltip(f, "Focal length in pixels Y. 0 = auto (image height).\n"
                   "In expand mode: this is K_new focal Y (from undistorted JSON).");

        Double_knob(f, &_centerX, "center_x", "Principal Point X");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Principal point X as fraction of image width (0.5 = center).\n"
                   "In expand mode: K_new principal point X (from undistorted JSON).");
        Double_knob(f, &_centerY, "center_y", "Principal Point Y");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Principal point Y, top-down fraction (0.5 = center).\n"
                   "In expand mode: K_new principal point Y (from undistorted JSON).");

        Divider(f, "Native (Calibration) Resolution");

        Int_knob(f, &_nativeW, "native_w", "Native Width");
        SetRange(f, 0, 8192);
        Tooltip(f, "Width of the calibration canvas (pixels).\n"
                   "When set, focal_x is scaled to the current output width:\n"
                   "  fx = focal_x * (output_w / native_w)\n"
                   "In expand mode: K_new canvas width (from undistorted JSON).\n"
                   "Set 0 to disable scaling.");
        Int_knob(f, &_nativeH, "native_h", "Native Height");
        SetRange(f, 0, 8192);
        Tooltip(f, "Height of the calibration canvas (pixels).\n"
                   "When set, focal_y is scaled to the current output height:\n"
                   "  fy = focal_y * (output_h / native_h)\n"
                   "In expand mode: K_new canvas height (from undistorted JSON).\n"
                   "Set 0 to disable scaling.");

        Divider(f, "Filtering");

        Enumeration_knob(f, &_filter, FILTER_NAMES, "filter", "Filter");
        Tooltip(f, "Pixel interpolation.\n"
                   "Nearest — fastest.  Bilinear — default.  Bicubic — best quality.");

        Double_knob(f, &_alpha, "alpha", "Alpha");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Fisheye mode only: balance parameter for\n"
                   "  cv::fisheye::estimateNewCameraMatrixForUndistortRectify.\n"
                   "  0 = crop to avoid black borders.\n"
                   "  1 = retain all source pixels (default).\n"
                   "Perspective mode: this knob has no effect\n"
                   "  (new_K equals the original K, matching Python K.copy() behaviour).");

        Divider(f, "Computed Output Matrix (new_K)");

        Double_knob(f, &_newFx, "new_fx", "new Focal X");
        SetFlags(f, Knob::READ_ONLY | Knob::NO_ANIMATION | Knob::NO_RERENDER);
        Tooltip(f, "Output focal length X.\n"
                   "Fisheye: computed by estimateNewCameraMatrixForUndistortRectify.\n"
                   "Perspective: equals focal_x (K.copy() behaviour).");
        Double_knob(f, &_newFy, "new_fy", "new Focal Y");
        SetFlags(f, Knob::READ_ONLY | Knob::NO_ANIMATION | Knob::NO_RERENDER);
        Tooltip(f, "Output focal length Y.");
        Double_knob(f, &_newCx, "new_cx", "new Principal X");
        SetFlags(f, Knob::READ_ONLY | Knob::NO_ANIMATION | Knob::NO_RERENDER);
        Tooltip(f, "Output principal point X (pixels).");
        Double_knob(f, &_newCy, "new_cy", "new Principal Y");
        SetFlags(f, Knob::READ_ONLY | Knob::NO_ANIMATION | Knob::NO_RERENDER);
        Tooltip(f, "Output principal point Y (pixels).");

        Divider(f, "NeRFStudio Import");

        File_knob(f, &_jsonFile, "json_file", "Primary JSON");
        Tooltip(f, "Standard mode: path to a single nerfstudio transforms.json.\n"
                   "Expand mode: path to the UNDISTORTED transforms_undistorted.json\n"
                   "  (K_new canvas geometry; distortion coefficients zeroed).\n"
                   "Fields loaded: fl_x/fl_y, k1\xe2\x80\x93k4, p1, p2,\n"
                   "cx/w, cy/h, w/h, is_fisheye.");

        File_knob(f, &_origJsonFile, "orig_json_file", "Original JSON");
        Tooltip(f, "Expand mode only: path to the ORIGINAL (distorted) transforms.json.\n"
                   "Corresponds to Python --original_json.\n"
                   "When set, Load button additionally fills K_orig parameters\n"
                   "and overrides distortion coefficients with real values.\n"
                   "Leave empty for standard (non-expand) mode.\n"
                   "Note: expand mode is perspective-only (not fisheye).");

        Button(f, "load_json", "Load from JSON(s)");
        Tooltip(f, "Load camera parameters from the JSON file(s) above.\n"
                   "Always loads Primary JSON first (K_new / focal / distortion).\n"
                   "If Original JSON is also set, loads it second:\n"
                   "  fills K_orig knobs and OVERRIDES distortion coefficients\n"
                   "  with the real values from the original camera.\n"
                   "Using a single button guarantees correct load order.");

        Divider(f, "Original Camera Parameters (Expand Mode)");

        Double_knob(f, &_origFocalX, "orig_focal_x", "Orig Focal X");
        SetRange(f, 0, 10000);
        Tooltip(f, "K_orig focal length X in pixels (at orig_w resolution).");
        Double_knob(f, &_origFocalY, "orig_focal_y", "Orig Focal Y");
        SetRange(f, 0, 10000);
        Tooltip(f, "K_orig focal length Y in pixels (at orig_h resolution).");

        Double_knob(f, &_origCenterX, "orig_center_x", "Orig Principal X");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "K_orig principal point X as fraction of orig_w (0.5 = center).");
        Double_knob(f, &_origCenterY, "orig_center_y", "Orig Principal Y");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "K_orig principal point Y as fraction of orig_h (0.5 = center).");

        Int_knob(f, &_origW, "orig_w", "Orig Width");
        SetRange(f, 0, 8192);
        Tooltip(f, "Original (distorted) canvas width in pixels.\n"
                   "Set > 0 to enable expand mode.");
        Int_knob(f, &_origH, "orig_h", "Orig Height");
        SetRange(f, 0, 8192);
        Tooltip(f, "Original (distorted) canvas height in pixels.\n"
                   "Set > 0 to enable expand mode.");
    }

    // ── knob_changed ──────────────────────────────────────────────────────────
    int knob_changed(Knob* k) override
    {
        if (k->is("load_json")) {
            // Always load primary JSON first, then original JSON (if set) second.
            // This guarantees K_orig and real distortion coefficients override
            // any zeroed values that may have come from the undistorted JSON.
            _loadFromJson();
            if (_origJsonFile && *_origJsonFile)
                _loadOrigJson();
            return 1;
        }
        return Iop::knob_changed(k);
    }

    // ── _validate ─────────────────────────────────────────────────────────────
    void _validate(bool for_real) override
    {
        copy_info();
        _isFirstTime = true;  // invalidate cache on any parameter change

        const bool expandMode = (_origW > 0 && _origH > 0);

        // In expand mode, change the output bounding box to the target canvas size.
        // Output scales proportionally if the input differs from the calibration size.
        //   Undistort: input is K_orig space  → output is K_new space
        //     scale = input / orig_ref  →  out = native_ref * scale
        //   Distort:   input is K_new space   → output is K_orig space
        //     scale = input / native_ref  →  out = orig_ref * scale
        if (expandMode) {
            const int inW = input(0)->info().w();
            const int inH = input(0)->info().h();
            int out_w, out_h;
            if (_mode == 0) {
                if (_origW > 0 && _nativeW > 0) {
                    const double sx = static_cast<double>(inW) / _origW;
                    const double sy = static_cast<double>(inH) / _origH;
                    out_w = static_cast<int>(std::round(_nativeW * sx));
                    out_h = static_cast<int>(std::round(_nativeH * sy));
                } else {
                    out_w = (_nativeW > 0) ? _nativeW : inW;
                    out_h = (_nativeH > 0) ? _nativeH : inH;
                }
            } else {
                if (_nativeW > 0 && _origW > 0) {
                    const double sx = static_cast<double>(inW) / _nativeW;
                    const double sy = static_cast<double>(inH) / _nativeH;
                    out_w = static_cast<int>(std::round(_origW * sx));
                    out_h = static_cast<int>(std::round(_origH * sy));
                } else {
                    out_w = (_origW > 0) ? _origW : inW;
                    out_h = (_origH > 0) ? _origH : inH;
                }
            }
            // Update both the format window (full canvas) and the data window.
            // _outputFmt is a class member so its lifetime outlasts _validate();
            // info_.format() stores a pointer, so a local Format would dangle.
            _outputFmt = Format(out_w, out_h, 1.0f);
            info_.format(_outputFmt);
            info_.set(0, 0, out_w, out_h);
        }

        const int w = info_.w();
        const int h = info_.h();
        if (w > 0 && h > 0) {
            double fx, fy;
            _effectiveFocal(w, h, fx, fy);
            const double cx = _centerX * w;
            const double cy = _centerY * h;
            const cv::Mat K = (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);

            cv::Mat newK;
            if (_isFisheye) {
                // Fisheye: compute new_K via estimateNewCameraMatrixForUndistortRectify.
                // This is pure matrix math (no parallel_for_), safe to call in Nuke.
                const cv::Mat D_fish = (cv::Mat_<double>(4,1) << _k1, _k2, _p1, _p2);
                cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
                    K, D_fish, cv::Size(w, h), cv::Mat::eye(3,3,CV_64F), newK, _alpha);
            } else {
                // Perspective: new_K = K (matches Python K.copy() behaviour).
                // alpha has no effect in perspective mode.
                newK = K.clone();
            }

            _newFx = newK.at<double>(0,0);
            _newFy = newK.at<double>(1,1);
            _newCx = newK.at<double>(0,2);
            _newCy = newK.at<double>(1,2);

            if (Knob* k = knob("new_fx")) k->set_value(_newFx);
            if (Knob* k = knob("new_fy")) k->set_value(_newFy);
            if (Knob* k = knob("new_cx")) k->set_value(_newCx);
            if (Knob* k = knob("new_cy")) k->set_value(_newCy);
        }
    }

    // ── _request ──────────────────────────────────────────────────────────────
    void _request(int /*x*/, int /*y*/, int /*r*/, int /*t*/,
                  ChannelMask channels, int count) override
    {
        // In expand mode the input dimensions differ from the output format.
        // Always request the full input frame using the *input* node's info.
        const Info& inInfo = input(0)->info();
        input(0)->request(0, 0, inInfo.w(), inInfo.h(), channels, count);
    }

    // ── engine ────────────────────────────────────────────────────────────────
    void engine(int y, int x, int r, ChannelMask channels, Row& row) override
    {
        const int outW = info_.w();
        const int outH = info_.h();

        if (y < 0 || y >= outH) {
            row.erase(channels);
            return;
        }

        // Build full-frame cache on first call (double-checked lock)
        if (_isFirstTime) {
            Guard guard(_lock);
            if (_isFirstTime) {
                _buildCache(outW, outH, channels);
                _isFirstTime = false;
            }
        }

        // Nuke row 0 = bottom; OpenCV row 0 = top
        const int cvRow = (outH - 1) - y;

        // Pass through any channels we didn't process
        ChannelSet passThrough = channels;
        foreach(chan, channels) {
            if (_outputCache.count(chan))
                passThrough -= chan;
        }
        if (passThrough)
            input(0)->get(y, x, r, passThrough, row);

        // Copy cached output
        foreach(chan, channels) {
            const auto it = _outputCache.find(chan);
            if (it == _outputCache.end()) continue;

            float* out = row.writable(chan);
            const float* src = it->second.ptr<float>(cvRow) + x;
            std::memcpy(out + x, src, (r - x) * sizeof(float));
        }
    }

private:
    // ── _effectiveFocal ───────────────────────────────────────────────────────
    // Returns (fx, fy) for K_new in pixels for the given output size (w x h).
    void _effectiveFocal(int w, int h, double& fx, double& fy) const
    {
        if (_focalX > 0.0) {
            const double scale = (_nativeW > 0)
                ? static_cast<double>(w) / _nativeW : 1.0;
            fx = _focalX * scale;
        } else {
            fx = static_cast<double>(w);
        }
        if (_focalY > 0.0) {
            const double scale = (_nativeH > 0)
                ? static_cast<double>(h) / _nativeH : 1.0;
            fy = _focalY * scale;
        } else {
            fy = static_cast<double>(h);
        }
    }

    // ── _effectiveOrigFocal ───────────────────────────────────────────────────
    // Returns (ofx, ofy, ocx, ocy) for K_orig scaled to the actual input size.
    void _effectiveOrigFocal(int inW, int inH,
                             double& ofx, double& ofy,
                             double& ocx, double& ocy) const
    {
        if (_origFocalX > 0.0) {
            const double sx = (_origW > 0) ? static_cast<double>(inW) / _origW : 1.0;
            const double sy = (_origH > 0) ? static_cast<double>(inH) / _origH : 1.0;
            ofx = _origFocalX * sx;
            ofy = _origFocalY * sy;
        } else {
            ofx = static_cast<double>(inW);
            ofy = static_cast<double>(inH);
        }
        ocx = _origCenterX * inW;
        ocy = _origCenterY * inH;
    }

    // ── _loadFromJson ─────────────────────────────────────────────────────────
    // Parses a nerfstudio transforms.json and updates camera parameter knobs.
    // In expand mode: load the UNDISTORTED JSON here (K_new canvas geometry).
    void _loadFromJson()
    {
        if (!_jsonFile || !*_jsonFile) return;

        std::ifstream ifs(_jsonFile);
        if (!ifs.is_open()) return;

        nlohmann::json j;
        try {
            j = nlohmann::json::parse(ifs);
        } catch (...) {
            return;
        }

        auto setK = [&](const char* knobName, const char* jsonKey) {
            auto it = j.find(jsonKey);
            if (it != j.end() && it->is_number()) {
                if (Knob* k = knob(knobName))
                    k->set_value(it->get<double>());
            }
        };

        setK("focal_x", "fl_x");
        setK("focal_y", "fl_y");
        setK("k1", "k1");
        setK("k2", "k2");
        setK("k3", "k3");
        setK("k4", "k4");
        setK("p1", "p1");
        setK("p2", "p2");

        // is_fisheye
        {
            auto it = j.find("is_fisheye");
            if (it != j.end() && it->is_boolean()) {
                if (Knob* k = knob("is_fisheye"))
                    k->set_value(it->get<bool>() ? 1.0 : 0.0);
            }
        }

        // Principal point: stored as pixel coords in JSON, plugin uses 0-1 fraction.
        const double w  = j.value("w",  0.0);
        const double h  = j.value("h",  0.0);
        const double cx = j.value("cx", 0.0);
        const double cy = j.value("cy", 0.0);
        if (w > 0.0) if (Knob* k = knob("center_x")) k->set_value(cx / w);
        if (h > 0.0) if (Knob* k = knob("center_y")) k->set_value(cy / h);

        // Store the calibration resolution for focal length scaling.
        if (w > 0.0) if (Knob* k = knob("native_w")) k->set_value(w);
        if (h > 0.0) if (Knob* k = knob("native_h")) k->set_value(h);
    }

    // ── _loadOrigJson ─────────────────────────────────────────────────────────
    // Parses the ORIGINAL (distorted) transforms.json.
    // Sets K_orig knobs (orig_focal_x/y, orig_center_x/y, orig_w/h) and
    // OVERRIDES the distortion coefficient knobs with the real values.
    void _loadOrigJson()
    {
        if (!_origJsonFile || !*_origJsonFile) return;

        std::ifstream ifs(_origJsonFile);
        if (!ifs.is_open()) return;

        nlohmann::json j;
        try {
            j = nlohmann::json::parse(ifs);
        } catch (...) {
            return;
        }

        auto setK = [&](const char* knobName, const char* jsonKey) {
            auto it = j.find(jsonKey);
            if (it != j.end() && it->is_number()) {
                if (Knob* k = knob(knobName))
                    k->set_value(it->get<double>());
            }
        };

        // K_orig intrinsics
        setK("orig_focal_x", "fl_x");
        setK("orig_focal_y", "fl_y");

        const double w  = j.value("w",  0.0);
        const double h  = j.value("h",  0.0);
        const double cx = j.value("cx", 0.0);
        const double cy = j.value("cy", 0.0);
        if (w > 0.0) if (Knob* k = knob("orig_center_x")) k->set_value(cx / w);
        if (h > 0.0) if (Knob* k = knob("orig_center_y")) k->set_value(cy / h);
        if (w > 0.0) if (Knob* k = knob("orig_w")) k->set_value(w);
        if (h > 0.0) if (Knob* k = knob("orig_h")) k->set_value(h);

        // Override distortion coefficients with real values from original JSON
        setK("k1", "k1");
        setK("k2", "k2");
        setK("k3", "k3");
        setK("k4", "k4");
        setK("p1", "p1");
        setK("p2", "p2");

        // is_fisheye from original JSON
        {
            auto it = j.find("is_fisheye");
            if (it != j.end() && it->is_boolean()) {
                if (Knob* k = knob("is_fisheye"))
                    k->set_value(it->get<bool>() ? 1.0 : 0.0);
            }
        }
    }

    // ── Sampling helpers (no cv::remap — avoids OpenCV thread pool) ───────────
    // cv::remap uses parallel_for_ internally; its thread pool conflicts with
    // Nuke 17's scheduler causing purecall / null-deref crashes. We apply the
    // precomputed maps ourselves in a plain single-threaded loop instead.

    static float _sampleNearest(const cv::Mat& src, float sx, float sy)
    {
        const int ix = static_cast<int>(sx + 0.5f);
        const int iy = static_cast<int>(sy + 0.5f);
        if (ix < 0 || ix >= src.cols || iy < 0 || iy >= src.rows) return 0.0f;
        return src.at<float>(iy, ix);
    }

    static float _sampleBilinear(const cv::Mat& src, float sx, float sy)
    {
        const int x0 = static_cast<int>(std::floor(sx));
        const int y0 = static_cast<int>(std::floor(sy));
        const float fx = sx - x0, fy = sy - y0;
        const int W = src.cols, H = src.rows;
        auto at = [&](int x, int y) -> float {
            if (x < 0 || x >= W || y < 0 || y >= H) return 0.0f;
            return src.at<float>(y, x);
        };
        return at(x0,   y0  ) * (1-fx) * (1-fy)
             + at(x0+1, y0  ) *    fx  * (1-fy)
             + at(x0,   y0+1) * (1-fx) *    fy
             + at(x0+1, y0+1) *    fx  *    fy;
    }

    static float _sampleBicubic(const cv::Mat& src, float sx, float sy)
    {
        // Catmull-Rom cubic kernel
        auto cubic = [](float t) -> float {
            t = std::abs(t);
            if (t < 1.0f) return 1.5f*t*t*t - 2.5f*t*t + 1.0f;
            if (t < 2.0f) return -0.5f*t*t*t + 2.5f*t*t - 4.0f*t + 2.0f;
            return 0.0f;
        };
        const int x0 = static_cast<int>(std::floor(sx));
        const int y0 = static_cast<int>(std::floor(sy));
        const int W = src.cols, H = src.rows;
        auto at = [&](int x, int y) -> float {
            x = std::max(0, std::min(W-1, x));
            y = std::max(0, std::min(H-1, y));
            return src.at<float>(y, x);
        };
        if (sx < -2 || sx >= W+1 || sy < -2 || sy >= H+1) return 0.0f;

        float result = 0.0f;
        for (int dy = -1; dy <= 2; ++dy)
            for (int dx = -1; dx <= 2; ++dx)
                result += at(x0+dx, y0+dy) * cubic(sx-(x0+dx)) * cubic(sy-(y0+dy));
        return result;
    }

    static void _applyMap(const cv::Mat& src, const cv::Mat& mapX,
                          const cv::Mat& mapY, int filter, cv::Mat& dst)
    {
        const int h = mapX.rows, w = mapX.cols;
        dst.create(h, w, CV_32FC1);
        for (int row = 0; row < h; ++row) {
            float*       pOut = dst .ptr<float>(row);
            const float* pX   = mapX.ptr<float>(row);
            const float* pY   = mapY.ptr<float>(row);
            for (int col = 0; col < w; ++col) {
                const float sx = pX[col], sy = pY[col];
                if (filter == 0)
                    pOut[col] = _sampleNearest (src, sx, sy);
                else if (filter == 2)
                    pOut[col] = _sampleBicubic (src, sx, sy);
                else
                    pOut[col] = _sampleBilinear(src, sx, sy);
            }
        }
    }

    // ── _buildCache ───────────────────────────────────────────────────────────
    // Called with _lock held.
    void _buildCache(int outW, int outH, ChannelMask channels)
    {
        _outputCache.clear();

        // In expand mode the input dimensions differ from the output.
        const int inW = input(0)->info().w();
        const int inH = input(0)->info().h();

        double fx, fy;
        _effectiveFocal(outW, outH, fx, fy);
        const double cx = _centerX * outW;
        const double cy = _centerY * outH;

        cv::Mat mapX, mapY;
        _buildMaps(outW, outH, inW, inH, fx, fy, cx, cy, mapX, mapY);

        // Read the full input frame via Tile (Nuke's standard full-frame accessor)
        Tile tile(input0(), 0, 0, inW, inH, channels);
        if (aborted()) return;

        foreach(chan, channels) {
            // Copy Tile into top-down cv::Mat (input dimensions)
            cv::Mat srcMat(inH, inW, CV_32FC1);
            for (int nukeRow = 0; nukeRow < inH; ++nukeRow) {
                const int cvRow = (inH - 1) - nukeRow;
                for (int col = 0; col < inW; ++col)
                    srcMat.at<float>(cvRow, col) = tile[chan][nukeRow][col];
            }

            // Output mat has output dimensions (mapX/Y are outH × outW)
            cv::Mat dstMat;
            _applyMap(srcMat, mapX, mapY, _filter, dstMat);

            _outputCache[chan] = std::move(dstMat);
        }
    }

    // ── _buildMaps ────────────────────────────────────────────────────────────
    // Builds pixel-space lookup maps (mapX, mapY) of size outH × outW.
    // Each entry maps: output pixel (col, row) → source pixel in input image.
    //
    // Standard mode (expandMode = false): K_new = K_orig = K.
    // Expand mode   (expandMode = true):  two different K matrices.
    //
    // Fisheye mode: equidistant OpenCV model, manual pixel loops (no parallel_for_).
    // Perspective:  Brown-Conrady rational model.
    void _buildMaps(int outW, int outH, int inW, int inH,
                    double fx, double fy, double cx, double cy,
                    cv::Mat& mapX, cv::Mat& mapY) const
    {
        const bool expandMode = (_origW > 0 && _origH > 0);

        // ── K_new (output/undistorted space) ──────────────────────────────────
        // nfx, nfy, ncx, ncy are the K_new parameters at the output resolution.
        // For perspective standard mode: K_new = K (K.copy() matching Python).
        // For fisheye: K_new computed by estimateNewCameraMatrixForUndistortRectify.
        // For expand mode: K_new comes from the primary JSON focal/center/native knobs
        //   already resolved to the output size by _effectiveFocal.
        double nfx = fx, nfy = fy, ncx = cx, ncy = cy;

        if (_isFisheye) {
            const cv::Mat K = (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
            const cv::Mat D_fish = (cv::Mat_<double>(4,1) << _k1, _k2, _p1, _p2);
            cv::Mat newK;
            cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
                K, D_fish, cv::Size(outW, outH),
                cv::Mat::eye(3,3,CV_64F), newK, _alpha);
            nfx = newK.at<double>(0,0);
            nfy = newK.at<double>(1,1);
            ncx = newK.at<double>(0,2);
            ncy = newK.at<double>(1,2);
        }

        // ── K_orig (input/distorted space) ────────────────────────────────────
        // Standard mode: K_orig = K_new (same K, nfx/nfy/ncx/ncy).
        // Expand mode:   K_orig from orig_* knobs, scaled to actual input size.
        double ofx, ofy, ocx, ocy;
        if (expandMode) {
            _effectiveOrigFocal(inW, inH, ofx, ofy, ocx, ocy);
        } else {
            ofx = nfx; ofy = nfy; ocx = ncx; ocy = ncy;
        }

        // ── Build maps ────────────────────────────────────────────────────────
        const cv::Size outSz(outW, outH);
        mapX.create(outSz, CV_32FC1);
        mapY.create(outSz, CV_32FC1);

        constexpr double EPS = 1e-8;

        for (int row = 0; row < outH; ++row) {
            float* pX = mapX.ptr<float>(row);
            float* pY = mapY.ptr<float>(row);

            for (int col = 0; col < outW; ++col) {

                if (_isFisheye) {
                    // ── Fisheye (no expand mode) ──────────────────────────────
                    // Fisheye coefficients: k1=_k1, k2=_k2, fisheye k3=_p1, fisheye k4=_p2
                    // (Matches Python D[:4] = [k1, k2, p1, p2] passed to cv2.fisheye.*)
                    const double k1f=_k1, k2f=_k2, k3f=_p1, k4f=_p2;

                    if (_mode == 0) {
                        // FISHEYE UNDISTORT
                        // Output pixel in K_new space → source pixel in K_orig space.
                        // Apply fisheye forward model to get the distorted direction.
                        const double xu = (col - ncx) / nfx;
                        const double yu = (row - ncy) / nfy;
                        const double r  = std::sqrt(xu*xu + yu*yu);
                        const double theta = std::atan(r);
                        const double t2 = theta*theta;
                        const double t4 = t2*t2, t6 = t4*t2, t8 = t6*t2;
                        const double theta_d = theta * (1.0 + k1f*t2 + k2f*t4 + k3f*t6 + k4f*t8);
                        const double scale = (r > EPS) ? theta_d / r : 1.0;
                        const double xd = xu * scale;
                        const double yd = yu * scale;
                        // Project through K_orig (= K_new in standard mode)
                        pX[col] = static_cast<float>(xd * ofx + ocx);
                        pY[col] = static_cast<float>(yd * ofy + ocy);

                    } else {
                        // FISHEYE DISTORT
                        // Output pixel in K_orig space → source pixel in K_new space.
                        // Newton-iterate: find theta from theta_d = r_d.
                        const double xd = (col - ocx) / ofx;
                        const double yd = (row - ocy) / ofy;
                        const double r_d = std::sqrt(xd*xd + yd*yd);
                        double theta = r_d;  // initial guess
                        for (int iter = 0; iter < 10; ++iter) {
                            const double t2 = theta*theta;
                            const double t4 = t2*t2, t6 = t4*t2, t8 = t6*t2;
                            const double f  = theta*(1.0 + k1f*t2 + k2f*t4 + k3f*t6 + k4f*t8) - r_d;
                            const double fp = 1.0 + 3.0*k1f*t2 + 5.0*k2f*t4 + 7.0*k3f*t6 + 9.0*k4f*t8;
                            const double delta = f / (fp != 0.0 ? fp : 1.0);
                            theta -= delta;
                            if (std::abs(delta) < 1e-10) break;
                        }
                        const double r_u = std::tan(theta);
                        const double scale = (r_d > EPS) ? r_u / r_d : 1.0;
                        const double xu = xd * scale;
                        const double yu = yd * scale;
                        // Project through K_new
                        pX[col] = static_cast<float>(xu * nfx + ncx);
                        pY[col] = static_cast<float>(yu * nfy + ncy);
                    }

                } else {
                    // ── Perspective (Brown-Conrady rational) ──────────────────

                    if (_mode == 0) {
                        // PERSPECTIVE UNDISTORT
                        // Output pixel in K_new space → source pixel in K_orig (input) space.
                        // Standard mode: K_new = K_orig so this is pure forward distortion.
                        // Expand mode:   K_new ≠ K_orig, maps between two different canvases.
                        const double xu = (col - ncx) / nfx;
                        const double yu = (row - ncy) / nfy;
                        const double r2 = xu*xu + yu*yu;
                        const double r4 = r2*r2, r6 = r4*r2;
                        const double denom = 1.0 + _k4*r2 + _k5*r4 + _k6*r6;
                        const double d = (1.0 + _k1*r2 + _k2*r4 + _k3*r6)
                                       / (denom != 0.0 ? denom : 1.0);
                        const double xd = xu*d + 2.0*_p1*xu*yu + _p2*(r2 + 2.0*xu*xu);
                        const double yd = yu*d + _p1*(r2 + 2.0*yu*yu) + 2.0*_p2*xu*yu;
                        pX[col] = static_cast<float>(xd * ofx + ocx);
                        pY[col] = static_cast<float>(yd * ofy + ocy);

                    } else {
                        // PERSPECTIVE DISTORT
                        // Output pixel in K_orig space → source pixel in K_new (input) space.
                        // Newton-iterate to invert the distortion model.
                        const double xd = (col - ocx) / ofx;
                        const double yd = (row - ocy) / ofy;
                        double xu = xd, yu = yd;
                        for (int iter = 0; iter < 10; ++iter) {
                            const double r2 = xu*xu + yu*yu;
                            const double r4 = r2*r2, r6 = r4*r2;
                            const double denom = 1.0 + _k4*r2 + _k5*r4 + _k6*r6;
                            const double d = (1.0 + _k1*r2 + _k2*r4 + _k3*r6)
                                           / (denom != 0.0 ? denom : 1.0);
                            const double ex = xd - (xu*d + 2.0*_p1*xu*yu + _p2*(r2 + 2.0*xu*xu));
                            const double ey = yd - (yu*d + _p1*(r2 + 2.0*yu*yu) + 2.0*_p2*xu*yu);
                            xu += ex;
                            yu += ey;
                        }
                        // Project undistorted coords through K_new to get source pixel
                        pX[col] = static_cast<float>(xu * nfx + ncx);
                        pY[col] = static_cast<float>(yu * nfy + ncy);
                    }
                }
            }
        }
    }

public:
    const char* Class()     const override { return CLASS; }
    const char* node_help() const override { return HELP;  }

    static const Iop::Description description;
    static Iop* build(Node* node) { return new LensDistort(node); }
};

// -----------------------------------------------------------------------------
const Iop::Description LensDistort::description(CLASS, "Filter/LensDistortCV", LensDistort::build);
