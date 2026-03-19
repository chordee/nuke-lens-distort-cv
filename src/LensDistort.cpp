// LensDistort.cpp
// Nuke NDK plugin: OpenCV-based lens distortion / undistortion
//
// Uses PlanarIop — lens distortion requires the full frame for cv::remap.
//
// Knobs:
//   mode     : Undistort / Distort
//   k1,k2,k3 : Radial distortion coefficients (Brown-Conrady)
//   p1, p2   : Tangential distortion coefficients
//   focal_x  : Focal length in pixels X  (0 = auto from image width)
//   focal_y  : Focal length in pixels Y  (0 = auto from image height)
//   center_x : Principal point X, normalized 0-1  (default 0.5)
//   center_y : Principal point Y, normalized 0-1  (default 0.5)
//   filter   : Nearest / Bilinear / Bicubic
//
// Build: see CMakeLists.txt
// Requires: Nuke 17 NDK, OpenCV 4.x (static or shared)

#include "DDImage/PlanarIop.h"
#include "DDImage/Knobs.h"
#include "DDImage/ImagePlane.h"
#include "DDImage/NukeWrapper.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <cstring>
#include <mutex>

using namespace DD::Image;

// -----------------------------------------------------------------------------
static const char* const CLASS = "LensDistort";
static const char* const HELP  =
    "<p><b>LensDistort</b> — OpenCV lens distortion / undistortion.</p>"
    "<p>Applies or removes radial + tangential lens distortion using the "
    "Brown-Conrady model (same convention as OpenCV / ShotCalibrate).</p>"
    "<p><b>Undistort</b> — removes distortion (straightens footage).<br>"
    "<b>Distort</b>   — adds distortion (match a real lens).</p>"
    "<p>Coefficients follow OpenCV convention:<br>"
    "k1, k2, k3 = radial &nbsp;&nbsp; p1, p2 = tangential</p>";

static const char* const MODE_NAMES[]   = { "Undistort", "Distort",  nullptr };
static const char* const FILTER_NAMES[] = { "Nearest",   "Bilinear", "Bicubic", nullptr };

// -----------------------------------------------------------------------------
class LensDistort : public PlanarIop
{
    // Knob storage
    int    _mode;
    double _k1, _k2, _k3;
    double _k4, _k5, _k6;
    double _p1, _p2;
    double _focalX, _focalY;
    double _centerX, _centerY;
    int    _filter;

    // Cached remap maps (guarded by _mapsMutex)
    mutable std::mutex _mapsMutex;
    cv::Mat _mapX, _mapY;

    // Fingerprint for dirty detection
    struct Params {
        double k1, k2, k3, k4, k5, k6, p1, p2, fx, fy, cx, cy;
        int mode, w, h;
        bool operator==(const Params& o) const {
            return k1==o.k1 && k2==o.k2 && k3==o.k3 &&
                   k4==o.k4 && k5==o.k5 && k6==o.k6 &&
                   p1==o.p1 && p2==o.p2 &&
                   fx==o.fx && fy==o.fy &&
                   cx==o.cx && cy==o.cy &&
                   mode==o.mode && w==o.w && h==o.h;
        }
    };
    Params _cached;
    bool   _mapsValid;

public:
    explicit LensDistort(Node* node)
        : PlanarIop(node)
        , _mode(0)
        , _k1(0), _k2(0), _k3(0)
        , _k4(0), _k5(0), _k6(0)
        , _p1(0), _p2(0)
        , _focalX(0), _focalY(0)
        , _centerX(0.5), _centerY(0.5)
        , _filter(1)
        , _mapsValid(false)
    {
        std::memset(&_cached, 0, sizeof(_cached));
        _cached.w = _cached.h = -1;
    }

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
        Tooltip(f, "Rational distortion k4 (denominator, r^2 term).\n"
                   "0 = disabled (equivalent to polynomial-only model).");

        Double_knob(f, &_k5, "k5", "k5");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Rational distortion k5 (denominator, r^4 term).\n"
                   "0 = disabled.");

        Double_knob(f, &_k6, "k6", "k6");
        SetRange(f, -2.0, 2.0);
        Tooltip(f, "Rational distortion k6 (denominator, r^6 term).\n"
                   "0 = disabled.");

        Double_knob(f, &_p1, "p1", "p1");
        SetRange(f, -0.5, 0.5);
        Tooltip(f, "Tangential distortion p1.");

        Double_knob(f, &_p2, "p2", "p2");
        SetRange(f, -0.5, 0.5);
        Tooltip(f, "Tangential distortion p2.");

        Divider(f, "Camera Intrinsics");

        Double_knob(f, &_focalX, "focal_x", "Focal X");
        SetRange(f, 0, 10000);
        Tooltip(f, "Focal length in pixels along X.\n"
                   "0 = auto-derive from image width.");

        Double_knob(f, &_focalY, "focal_y", "Focal Y");
        SetRange(f, 0, 10000);
        Tooltip(f, "Focal length in pixels along Y.\n"
                   "0 = auto-derive from image height.");

        Double_knob(f, &_centerX, "center_x", "Principal Point X");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Principal point X as fraction of image width "
                   "(0 = left edge, 1 = right edge).\n"
                   "0.5 = image center (default).\n"
                   "Matches OpenCV/ShotCalibrate convention: cx / image_width.");

        Double_knob(f, &_centerY, "center_y", "Principal Point Y");
        SetRange(f, 0.0, 1.0);
        Tooltip(f, "Principal point Y as fraction of image height, "
                   "measured from the TOP (0 = top edge, 1 = bottom edge).\n"
                   "0.5 = image center (default).\n"
                   "Matches OpenCV/ShotCalibrate convention: cy / image_height.\n"
                   "Note: this is top-down, opposite to Nuke's native Y direction.");

        Divider(f, "Filtering");

        Enumeration_knob(f, &_filter, FILTER_NAMES, "filter", "Filter");
        Tooltip(f, "Pixel interpolation method used by cv::remap.\n"
                   "Nearest  — fastest, blocky at large distortions.\n"
                   "Bilinear — good quality/speed balance (default).\n"
                   "Bicubic  — best quality, ~4x slower than bilinear.");
    }

    // ── _validate ─────────────────────────────────────────────────────────────
    void _validate(bool for_real) override
    {
        copy_info();  // pass through format, channels, bbox from input

        // Invalidate maps if format or params have changed
        const Format& fmt = input0().format();
        Params p;
        p.k1 = _k1; p.k2 = _k2; p.k3 = _k3;
        p.k4 = _k4; p.k5 = _k5; p.k6 = _k6;
        p.p1 = _p1; p.p2 = _p2;
        p.fx = _focalX; p.fy = _focalY;
        p.cx = _centerX; p.cy = _centerY;
        p.mode = _mode;
        p.w = fmt.width();
        p.h = fmt.height();

        {
            std::lock_guard<std::mutex> lock(_mapsMutex);
            if (!(_cached == p))
                _mapsValid = false;
        }
    }

    // ── getRequests ───────────────────────────────────────────────────────────
    void getRequests(const Box& /*box*/, const ChannelSet& channels,
                     int count, RequestOutput& reqData) const override
    {
        // Remap can sample any input pixel — always request the full format
        input0().request(input0().format(), channels, count);
    }

    // ── buildMaps ─────────────────────────────────────────────────────────────
    // Builds the cv::remap source maps.
    // Maps are in OpenCV coordinate space (top-down, pixel coords).
    // Y-flip between Nuke (bottom-up) and OpenCV (top-down) is handled
    // in renderStripe when copying data in/out.
    void buildMaps(int w, int h)
    {
        const double fx = (_focalX > 0.0) ? _focalX : static_cast<double>(w);
        const double fy = (_focalY > 0.0) ? _focalY : static_cast<double>(h);
        const double cx = _centerX * w;
        const double cy = _centerY * h;  // note: top-down after Y-flip in copy

        const cv::Mat K = (cv::Mat_<double>(3, 3)
            << fx,  0, cx,
                0, fy, cy,
                0,  0,  1);

        // OpenCV dist coeff order: k1, k2, p1, p2, k3[, k4, k5, k6]
        // 8-element form enables the rational model; k4=k5=k6=0 degrades to
        // the standard polynomial model with no change in result.
        const cv::Mat dist = (cv::Mat_<double>(1, 8)
            << _k1, _k2, _p1, _p2, _k3, _k4, _k5, _k6);
        const cv::Size sz(w, h);

        if (_mode == 0)
        {
            // ── Undistort ──────────────────────────────────────────────────
            // alpha=0: crop to only valid (non-black) pixels
            const cv::Mat newK = cv::getOptimalNewCameraMatrix(K, dist, sz, 0.0, sz);
            cv::initUndistortRectifyMap(K, dist, cv::Mat(), newK,
                                        sz, CV_32FC1, _mapX, _mapY);
        }
        else
        {
            // ── Distort ────────────────────────────────────────────────────
            // OpenCV has no initDistortMap; compute analytically.
            //
            // For each output pixel (col, row) treated as an undistorted
            // coordinate, apply the forward Brown-Conrady model to find the
            // distorted source coordinate to sample from.
            //
            //   xn = (col - cx) / fx
            //   yn = (row - cy) / fy
            //   r2 = xn^2 + yn^2
            //   radial = (1 + k1*r2 + k2*r4 + k3*r6)
            //          / (1 + k4*r2 + k5*r4 + k6*r6)
            //   xd = xn*radial + 2*p1*xn*yn + p2*(r2 + 2*xn^2)
            //   yd = yn*radial + p1*(r2 + 2*yn^2) + 2*p2*xn*yn
            //   map_x[row][col] = xd*fx + cx
            //   map_y[row][col] = yd*fy + cy
            //
            // When k4=k5=k6=0 the denominator is 1 and the rational model
            // reduces to the standard polynomial model.

            _mapX.create(sz, CV_32FC1);
            _mapY.create(sz, CV_32FC1);

            for (int row = 0; row < h; ++row)
            {
                float* pX = _mapX.ptr<float>(row);
                float* pY = _mapY.ptr<float>(row);

                for (int col = 0; col < w; ++col)
                {
                    const double xn = (col - cx) / fx;
                    const double yn = (row - cy) / fy;

                    const double r2 = xn*xn + yn*yn;
                    const double r4 = r2 * r2;
                    const double r6 = r4 * r2;

                    const double numer = 1.0 + _k1*r2 + _k2*r4 + _k3*r6;
                    const double denom = 1.0 + _k4*r2 + _k5*r4 + _k6*r6;
                    const double radial = numer / denom;
                    const double xd = xn*radial + 2.0*_p1*xn*yn + _p2*(r2 + 2.0*xn*xn);
                    const double yd = yn*radial + _p1*(r2 + 2.0*yn*yn) + 2.0*_p2*xn*yn;

                    pX[col] = static_cast<float>(xd * fx + cx);
                    pY[col] = static_cast<float>(yd * fy + cy);
                }
            }
        }

        // Update fingerprint
        _cached.k1 = _k1; _cached.k2 = _k2; _cached.k3 = _k3;
        _cached.k4 = _k4; _cached.k5 = _k5; _cached.k6 = _k6;
        _cached.p1 = _p1; _cached.p2 = _p2;
        _cached.fx = _focalX; _cached.fy = _focalY;
        _cached.cx = _centerX; _cached.cy = _centerY;
        _cached.mode = _mode;
        _cached.w = w; _cached.h = h;
        _mapsValid = true;
    }

    // ── renderStripe ──────────────────────────────────────────────────────────
    void renderStripe(ImagePlane& outputPlane) override
    {
        const int w = input0().format().width();
        const int h = input0().format().height();

        {
            std::lock_guard<std::mutex> lock(_mapsMutex);
            if (!_mapsValid)
                buildMaps(w, h);
        }

        // Fetch the full input frame
        ImagePlane inputPlane(input0().info(), false, outputPlane.channels());
        input0().fetchPlane(inputPlane);

        const int interp = (_filter == 0) ? cv::INTER_NEAREST
                         : (_filter == 2) ? cv::INTER_CUBIC
                                          : cv::INTER_LINEAR;

        // Process each channel (Nuke stores planar float32)
        for (Channel chan : outputPlane.channels())
        {
            // Copy input channel into cv::Mat with Y-flip
            // (Nuke row 0 = bottom; OpenCV row 0 = top)
            cv::Mat srcMat(h, w, CV_32FC1);
            for (int nukeRow = 0; nukeRow < h; ++nukeRow)
            {
                const int cvRow = (h - 1) - nukeRow;
                const float* src = inputPlane.readable().getRowPtr(nukeRow, chan);
                std::memcpy(srcMat.ptr<float>(cvRow), src, w * sizeof(float));
            }

            // Apply remap
            cv::Mat dstMat;
            cv::remap(srcMat, dstMat, _mapX, _mapY, interp,
                      cv::BORDER_CONSTANT, cv::Scalar(0.0f));

            // Copy result back to output stripe with Y-flip.
            // Respect stripe bounds: only copy the stripe's x-extent to avoid
            // writing past the allocated row buffer.
            const Box& stripe = outputPlane.bounds();
            const int  x0     = stripe.x();
            const int  copyW  = stripe.r() - x0;
            for (int nukeRow = stripe.y(); nukeRow < stripe.t(); ++nukeRow)
            {
                const int cvRow = (h - 1) - nukeRow;
                if (cvRow < 0 || cvRow >= h) continue;

                float* dst = outputPlane.writable().getRowPtr(nukeRow, chan);
                std::memcpy(dst, dstMat.ptr<float>(cvRow) + x0,
                            copyW * sizeof(float));
            }
        }
    }

    // ── Node metadata ─────────────────────────────────────────────────────────
    const char* Class()     const override { return CLASS; }
    const char* node_help() const override { return HELP;  }

    static const Iop::Description desc;
    static Iop* build(Node* node) { return new NukeWrapper(new LensDistort(node)); }
};

// -----------------------------------------------------------------------------
const Iop::Description LensDistort::desc(CLASS, "Filter/LensDistort", LensDistort::build);
