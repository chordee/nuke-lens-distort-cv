// LensDistort.cpp
// Nuke NDK plugin: OpenCV-based lens distortion / undistortion
//
// Pattern: Iop + engine() + Tile — matches SimpleBlurCached NDK example.
// No NukeWrapper (that is only for PixelIop / pixel_engine pattern).
// Full frame is computed on first engine() call and cached per-channel.
//
// Knobs:
//   mode     : Undistort / Distort
//   k1–k6    : Radial distortion coefficients (Brown-Conrady rational model)
//   p1, p2   : Tangential distortion coefficients
//   focal_x  : Focal length in pixels X  (0 = auto from image width)
//   focal_y  : Focal length in pixels Y  (0 = auto from image height)
//   center_x : Principal point X, normalized 0-1  (default 0.5)
//   center_y : Principal point Y, normalized 0-1  (default 0.5)
//   filter   : Nearest / Bilinear / Bicubic

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

#include <opencv2/core.hpp>

#include <nlohmann/json.hpp>

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
    "k1–k6 = radial &nbsp;&nbsp; p1, p2 = tangential</p>";

static const char* const MODE_NAMES[]   = { "Undistort", "Distort",  nullptr };
static const char* const FILTER_NAMES[] = { "Nearest",   "Bilinear", "Bicubic", nullptr };

// -----------------------------------------------------------------------------
class LensDistort : public Iop
{
    int    _mode;
    double _k1, _k2, _k3;
    double _k4, _k5, _k6;
    double _p1, _p2;
    double _focalX, _focalY;
    double _centerX, _centerY;
    int    _filter;

    // NeRFStudio JSON import
    const char* _jsonFile;

    // Full-frame output cache, one cv::Mat per channel
    Lock                     _lock;
    bool                     _isFirstTime;
    std::map<Channel, cv::Mat> _outputCache;   // top-down float32

public:
    explicit LensDistort(Node* node)
        : Iop(node)
        , _mode(0)
        , _k1(0), _k2(0), _k3(0)
        , _k4(0), _k5(0), _k6(0)
        , _p1(0), _p2(0)
        , _focalX(0), _focalY(0)
        , _centerX(0.5), _centerY(0.5)
        , _filter(1)
        , _jsonFile("")
        , _isFirstTime(true)
    {}

    // ── Knobs ─────────────────────────────────────────────────────────────────
    void knobs(Knob_Callback f) override
    {
        Enumeration_knob(f, &_mode, MODE_NAMES, "mode", "Mode");
        Tooltip(f, "Undistort removes distortion. Distort adds it.");

        Divider(f, "Distortion Coefficients");

        Double_knob(f, &_k1, "k1", "k1");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Radial distortion k1.");
        Double_knob(f, &_k2, "k2", "k2");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Radial distortion k2.");
        Double_knob(f, &_k3, "k3", "k3");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Radial distortion k3 (numerator, r^6 term).");
        Double_knob(f, &_k4, "k4", "k4");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Rational denominator k4 (r^2). 0 = disabled.");
        Double_knob(f, &_k5, "k5", "k5");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Rational denominator k5 (r^4). 0 = disabled.");
        Double_knob(f, &_k6, "k6", "k6");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Rational denominator k6 (r^6). 0 = disabled.");
        Double_knob(f, &_p1, "p1", "p1");
        SetRange(f, -0.5, 0.5);
        Tooltip(f, "Tangential distortion p1.");
        Double_knob(f, &_p2, "p2", "p2");
        SetRange(f, -0.5, 0.5);
        Tooltip(f, "Tangential distortion p2.");

        Divider(f, "Camera Intrinsics");

        Double_knob(f, &_focalX, "focal_x", "Focal X");
        SetRange(f, 0, 10000);
        Tooltip(f, "Focal length in pixels X. 0 = auto (image width).");
        Double_knob(f, &_focalY, "focal_y", "Focal Y");
        SetRange(f, 0, 10000);
        Tooltip(f, "Focal length in pixels Y. 0 = auto (image height).");

        Double_knob(f, &_centerX, "center_x", "Principal Point X");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Principal point X as fraction of image width (0.5 = center).\n"
                   "Matches OpenCV/ShotCalibrate convention: cx / image_width.");
        Double_knob(f, &_centerY, "center_y", "Principal Point Y");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Principal point Y, top-down fraction (0.5 = center).\n"
                   "Matches OpenCV convention: cy / image_height (top-down).\n"
                   "Note: opposite to Nuke's native bottom-up Y direction.");

        Divider(f, "Filtering");

        Enumeration_knob(f, &_filter, FILTER_NAMES, "filter", "Filter");
        Tooltip(f, "Pixel interpolation for cv::remap.\n"
                   "Nearest — fastest.  Bilinear — default.  Bicubic — best quality.");

        Divider(f, "NeRFStudio Import");

        File_knob(f, &_jsonFile, "json_file", "JSON File");
        Tooltip(f, "Path to a nerfstudio transforms.json file.\n"
                   "Fields mapped: fl_x/fl_y → Focal X/Y,\n"
                   "k1–k4, p1, p2 → distortion coefficients,\n"
                   "cx/w, cy/h → Principal Point X/Y.");
        Button(f, "load_json", "Load from JSON");
        Tooltip(f, "Read the JSON file above and fill in all camera parameters.");
    }

    // ── knob_changed ──────────────────────────────────────────────────────────
    int knob_changed(Knob* k) override
    {
        if (k->is("load_json")) {
            _loadFromJson();
            return 1;
        }
        return Iop::knob_changed(k);
    }

    // ── _validate ─────────────────────────────────────────────────────────────
    void _validate(bool for_real) override
    {
        copy_info();
        _isFirstTime = true;  // invalidate cache on any parameter change
    }

    // ── _request ──────────────────────────────────────────────────────────────
    void _request(int /*x*/, int /*y*/, int /*r*/, int /*t*/,
                  ChannelMask channels, int count) override
    {
        // Remap can sample any input pixel — request the full frame
        input(0)->request(0, 0, info_.w(), info_.h(), channels, count);
    }

    // ── engine ────────────────────────────────────────────────────────────────
    void engine(int y, int x, int r, ChannelMask channels, Row& row) override
    {
        const int w = info_.w();
        const int h = info_.h();

        if (y < 0 || y >= h) {
            row.erase(channels);
            return;
        }

        // Build full-frame cache on first call (double-checked lock)
        if (_isFirstTime) {
            Guard guard(_lock);
            if (_isFirstTime) {
                _buildCache(w, h, channels);
                _isFirstTime = false;
            }
        }

        // Nuke row 0 = bottom; OpenCV row 0 = top
        const int cvRow = (h - 1) - y;

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
    // ── _loadFromJson ─────────────────────────────────────────────────────────
    // Parses a nerfstudio transforms.json and updates camera parameter knobs.
    // JSON convention (instant-ngp / nerfstudio):
    //   fl_x, fl_y          → focal_x, focal_y  (pixels)
    //   k1, k2, k3, k4      → k1–k4  (OpenCV rational model)
    //   p1, p2              → p1, p2 (tangential)
    //   cx / w, cy / h      → center_x, center_y (top-down fraction)
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

        // Helper: set a Double knob by name if the JSON key is a number.
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

        // Principal point: stored as pixel coords in JSON, plugin uses 0-1 fraction.
        const double w  = j.value("w",  0.0);
        const double h  = j.value("h",  0.0);
        const double cx = j.value("cx", 0.0);
        const double cy = j.value("cy", 0.0);
        if (w > 0.0) if (Knob* k = knob("center_x")) k->set_value(cx / w);
        if (h > 0.0) if (Knob* k = knob("center_y")) k->set_value(cy / h);
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
            // Clamp to border for out-of-range (matches BORDER_REPLICATE for
            // edge pixels; center of frame is never out of range anyway).
            x = std::max(0, std::min(W-1, x));
            y = std::max(0, std::min(H-1, y));
            return src.at<float>(y, x);
        };
        // But return 0 if the sample point is well outside the image
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
        const int h = src.rows, w = src.cols;
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
    // Called with _lock held. Reads full input via Tile, applies remap maps.
    void _buildCache(int w, int h, ChannelMask channels)
    {
        _outputCache.clear();

        const double fx = (_focalX > 0.0) ? _focalX : static_cast<double>(w);
        const double fy = (_focalY > 0.0) ? _focalY : static_cast<double>(h);
        const double cx = _centerX * w;
        const double cy = _centerY * h;

        cv::Mat mapX, mapY;
        _buildMaps(w, h, fx, fy, cx, cy, mapX, mapY);

        // Read the full input frame via Tile (Nuke's standard full-frame accessor)
        Tile tile(input0(), 0, 0, w, h, channels);
        if (aborted()) return;

        foreach(chan, channels) {
            // Copy Tile into top-down cv::Mat
            cv::Mat srcMat(h, w, CV_32FC1);
            for (int nukeRow = 0; nukeRow < h; ++nukeRow) {
                const int cvRow = (h - 1) - nukeRow;
                for (int col = 0; col < w; ++col)
                    srcMat.at<float>(cvRow, col) = tile[chan][nukeRow][col];
            }

            cv::Mat dstMat;
            _applyMap(srcMat, mapX, mapY, _filter, dstMat);

            _outputCache[chan] = std::move(dstMat);
        }
    }

    // ── _buildMaps ────────────────────────────────────────────────────────────
    // Pure C++ — no OpenCV API calls (avoids parallel_for_ / thread pool).
    //
    // Undistort (mode=0):
    //   Each output pixel (col,row) maps from the undistorted grid.
    //   Apply forward Brown-Conrady to get the distorted source coord.
    //
    // Distort (mode=1):
    //   Each output pixel (col,row) is in distorted space.
    //   We need the undistorted source coord → iterative Newton inverse.
    void _buildMaps(int w, int h, double fx, double fy, double cx, double cy,
                    cv::Mat& mapX, cv::Mat& mapY) const
    {
        mapX.create(cv::Size(w, h), CV_32FC1);
        mapY.create(cv::Size(w, h), CV_32FC1);

        for (int row = 0; row < h; ++row) {
            float* pX = mapX.ptr<float>(row);
            float* pY = mapY.ptr<float>(row);

            for (int col = 0; col < w; ++col) {
                if (_mode == 0) {
                    // UNDISTORT: output is undistorted space.
                    // Forward model: given undistorted (xu,yu) → distorted (xd,yd).
                    // mapX/mapY = distorted source coords to sample from.
                    const double xu = (col - cx) / fx;
                    const double yu = (row - cy) / fy;
                    const double r2 = xu*xu + yu*yu;
                    const double r4 = r2*r2, r6 = r4*r2;
                    const double denom = 1.0 + _k4*r2 + _k5*r4 + _k6*r6;
                    const double d = (1.0 + _k1*r2 + _k2*r4 + _k3*r6)
                                   / (denom != 0.0 ? denom : 1.0);
                    const double xd = xu*d + 2.0*_p1*xu*yu + _p2*(r2 + 2.0*xu*xu);
                    const double yd = yu*d + _p1*(r2 + 2.0*yu*yu) + 2.0*_p2*xu*yu;
                    pX[col] = static_cast<float>(xd * fx + cx);
                    pY[col] = static_cast<float>(yd * fy + cy);
                } else {
                    // DISTORT: output is distorted space.
                    // We want the undistorted source coord (xu,yu) such that
                    // forward_model(xu,yu) = (xd,yd) for this output pixel.
                    // Solved iteratively (10 Newton steps — converges in 3-4).
                    const double xd = (col - cx) / fx;
                    const double yd = (row - cy) / fy;
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
                    pX[col] = static_cast<float>(xu * fx + cx);
                    pY[col] = static_cast<float>(yu * fy + cy);
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
