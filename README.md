# LensDistortCV — Nuke NDK Plugin

OpenCV-based lens distortion / undistortion node for Nuke 15+.

## Features

- **Undistort** — removes lens distortion (straighten footage)
- **Distort** — adds lens distortion (match a real lens)
- Full **Brown-Conrady** rational model: k1–k6, p1, p2
- **Fisheye model** — OpenCV equidistant (equidistant) fisheye: k1/k2 + fisheye k3/k4 (via p1/p2 knobs)
- Configurable focal length and principal point
- **Native resolution** knobs — focal lengths auto-scale when applying calibration data to a different resolution
- **Expand mode** — two-JSON workflow that maps between two different canvas sizes (e.g. original distorted ↔ undistorted expanded canvas)
- **Alpha** knob — fisheye balance parameter for `estimateNewCameraMatrixForUndistortRectify`
- Computed **new\_K** display — shows the effective output camera matrix
- Nearest / Bilinear / Bicubic filter modes
- Remap maps cached and only rebuilt when parameters change
- **NeRFStudio JSON import** — load camera intrinsics from one or two `transforms.json` files
- Self-contained: OpenCV statically linked into the DLL/SO (no runtime dependency)
- Matches output of Python `cv2.initUndistortRectifyMap` / `cv2.undistortPoints` / `cv2.fisheye.*` exactly

## Coefficients convention

Matches OpenCV / ShotCalibrate / NeRFStudio / most calibration tools:

```text
k1, k2, k3  — radial numerator   (polynomial model)
k4, k5, k6  — radial denominator (rational model, set to 0 to disable)
p1, p2      — tangential distortion

dist = [k1, k2, p1, p2, k3, k4, k5, k6]
```

The rational model formula:

```text
radial = (1 + k1·r² + k2·r⁴ + k3·r⁶)
       / (1 + k4·r² + k5·r⁴ + k6·r⁶)
```

k4, k5, k6 default to 0, which makes the denominator 1 and reduces the
model to the standard polynomial form — identical behaviour to a
5-coefficient calibration.

## Fisheye model

When **Fisheye Model** is enabled, the plugin uses OpenCV's equidistant fisheye model:

```text
theta_d = theta × (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
```

Coefficient mapping from knobs to fisheye model:

| Knob | Fisheye role |
| --- | --- |
| k1 | fisheye k1 |
| k2 | fisheye k2 |
| p1 | fisheye k3 |
| p2 | fisheye k4 |
| k3–k6 | ignored |

The **Alpha** knob controls `estimateNewCameraMatrixForUndistortRectify`:

| Alpha | Effect |
| --- | --- |
| `0.0` | Crop to avoid black borders |
| `1.0` | Retain all source pixels; corners may be black (default) |

In **perspective mode**, Alpha has no effect — the output camera matrix equals
the input K directly (matching Python's `K.copy()` behaviour).

The **Computed Output Matrix (new\_K)** section shows `new_fx`, `new_fy`,
`new_cx`, `new_cy` — the effective intrinsics of the output image, useful
for downstream 3D projection or matchmove work.

---

## Build: Windows

OpenCV is managed by **vcpkg** via a custom triplet that pins the exact MSVC
toolset. This is required because Nuke's DDImage ABI must match the compiler
used to build the plugin.

| Nuke version | Visual Studio | Preset |
| --- | --- | --- |
| 15.x | VS 2019 (v142) | `windows-vs2019` |
| 16.x / 17.x | VS 2022 (v143) | `windows-vs2022` |

### Prerequisites

| Tool | Version |
| --- | --- |
| Visual Studio 2019 | for Nuke 15 |
| Visual Studio 2022 | for Nuke 16/17 |
| CMake | 3.20+ |
| vcpkg | latest |
| Nuke | target version (for NDK headers + DDImage.lib) |

### Step 1 — Install vcpkg (once)

```powershell
git clone https://github.com/microsoft/vcpkg.git D:/vcpkg
D:/vcpkg/bootstrap-vcpkg.bat
```

### Step 2 — Set VCPKG\_ROOT (once)

```powershell
[System.Environment]::SetEnvironmentVariable("VCPKG_ROOT", "D:/vcpkg", "User")
# Restart your terminal after this
```

### Step 3 — Install OpenCV (once per toolset)

vcpkg packages must be pre-installed into separate directories — one per
toolset — because each preset uses `VCPKG_MANIFEST_INSTALL=OFF` to avoid
running vcpkg again during CMake configure.

```powershell
cd nuke-lens-distort-cv

# For Nuke 15 (VS2019 / v142)
vcpkg install --triplet x64-windows-static-md-v142 `
              --overlay-triplets triplets `
              --x-install-root vcpkg_installed_vs2019

# For Nuke 16/17 (VS2022 / v143)
vcpkg install --triplet x64-windows-static-md-v143 `
              --overlay-triplets triplets `
              --x-install-root vcpkg_installed_vs2022
```

> **Note:** The first run compiles OpenCV from source (~2–3 min per toolset).
> Subsequent runs restore from the local binary cache in seconds.

### Step 4 — Configure

`NUKE_ROOT` must be passed explicitly. The preset selects the compiler and
vcpkg installed directory automatically.

```powershell
# Nuke 15
cmake --preset windows-vs2019 -DNUKE_ROOT="C:/Program Files/Nuke15.1v5"

# Nuke 16/17
cmake --preset windows-vs2022 -DNUKE_ROOT="C:/Program Files/Nuke17.0v1"
```

### Step 5 — Build

```powershell
cmake --build --preset windows-vs2019
# or
cmake --build --preset windows-vs2022
```

Output:

- `build_vs2019\deploy\Release\LensDistortCV.dll`
- `build_vs2022\deploy\Release\LensDistortCV.dll`

### Step 6 — Install

Choose **one** of the following methods.

---

#### Method A — Personal install (recommended for single user)

```powershell
# Nuke 15
Copy-Item build_vs2019\deploy\Release\LensDistortCV.dll "$env:USERPROFILE\.nuke\"

# Nuke 16/17
Copy-Item build_vs2022\deploy\Release\LensDistortCV.dll "$env:USERPROFILE\.nuke\"
```

---

#### Method B — Studio shared install (multiple users, network path)

```powershell
[System.Environment]::SetEnvironmentVariable(
    "NUKE_PATH", "\\server\nuke_plugins", "Machine")

Copy-Item build_vs2022\deploy\Release\LensDistortCV.dll \\server\nuke_plugins\
```

`NUKE_PATH` accepts multiple paths separated by semicolons (`;`).

---

#### Method C — System install into the Nuke directory (requires admin)

```powershell
# Run PowerShell as Administrator
cmake --install build_vs2022 --prefix "C:\Program Files\Nuke17.0v1"
```

---

## Build: Linux

**Required compiler: GCC 11.2.1** (VFX Platform CY2022+, mandatory for Nuke 15+).
The preset enforces `gcc-11` / `g++-11` — configure will fail fast if not installed.

```bash
# Install GCC 11 if needed (Ubuntu/Debian)
sudo apt install gcc-11 g++-11

# Install vcpkg (once)
git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg
/opt/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=/opt/vcpkg   # add to ~/.bashrc

# Install OpenCV (once)
cd nuke-lens-distort-cv
vcpkg install --triplet x64-linux --x-install-root vcpkg_installed_linux

# Configure and build
cmake --preset linux -DNUKE_ROOT=/usr/local/Nuke17.0v1
cmake --build --preset linux
```

Output: `build_linux/deploy/LensDistortCV.so`

### Install (Linux)

```bash
# Personal
mkdir -p ~/.nuke && cp build_linux/deploy/LensDistortCV.so ~/.nuke/

# Studio shared
export NUKE_PATH="/studio/nuke_plugins:$NUKE_PATH"
cp build_linux/deploy/LensDistortCV.so /studio/nuke_plugins/

# System (requires sudo)
sudo cmake --install build_linux --prefix /usr/local/Nuke17.0v1
```

---

## Usage in Nuke

1. Launch Nuke — the plugin loads automatically from your plugin path
2. Create node: **Tab** → type `LensDistortCV`
3. Or find it in the menu: **Filter → LensDistortCV**

### Typical workflow

**Undistort footage from a calibrated camera:**

```text
k1 = -0.28   (from calibration)
k2 =  0.07
k3 =  0.0
k4–k6 = 0.0  (leave at 0 unless using rational model)
p1 =  0.001
p2 = -0.0005
focal_x = 0  (auto — uses image width)
focal_y = 0  (auto — uses image height)
center_x = 0.5
center_y = 0.5
native_w = 0  (0 = disabled; set to calibration width if focal_x is explicit)
native_h = 0  (0 = disabled; set to calibration height if focal_y is explicit)
alpha = 1.0  (retain all pixels; set to 0 to crop black borders)
Mode = Undistort
```

**Applying calibration data to a different resolution:**

If the coefficients were measured at 1280×720 but your plate is 2560×1440,
set `native_w = 1280` and `native_h = 720`. The plugin scales the focal
lengths automatically (`fx = focal_x × current_w / native_w`). Principal
point (`center_x` / `center_y`) is stored as a 0–1 fraction and scales
correctly on its own.

**Re-distort CG to match plate:**
Set the same coefficients, switch Mode to **Distort**.
The input image should be in the undistorted (new\_K) space —
i.e. the output of a previous Undistort pass with the same alpha.

### NeRFStudio JSON import

Camera parameters can be loaded from `transforms.json` files produced by
NeRFStudio, instant-ngp, or the
[AI\_colmap\_camera\_tracking](https://github.com/chordee/AI_colmap_camera_tracking)
pipeline. Manual entry of all knobs is equally supported — the JSON buttons
are a convenience shortcut.

#### Standard mode (single JSON)

1. Set **Original JSON** to your `transforms.json`
2. Click **Load from JSON(s)**

All camera parameters are filled in automatically:

| JSON field | Plugin knob | Conversion |
| --- | --- | --- |
| `fl_x` | Focal X | direct |
| `fl_y` | Focal Y | direct |
| `k1`, `k2` | k1, k2 | direct |
| `k3`, `k4` | perspective: k3, k4 · **fisheye: p1, p2** | routed by camera model |
| `p1`, `p2` | p1, p2 | perspective only (ignored for fisheye) |
| `camera_model` / `is_fisheye` | Fisheye Model | `OPENCV_FISHEYE` → on |
| `cx / w` | Principal Point X | normalised to 0–1 |
| `cy / h` | Principal Point Y | normalised to 0–1 |
| `w` | Native Width | calibration image width (pixels) |
| `h` | Native Height | calibration image height (pixels) |

#### Expand mode (two JSONs)

Some workflows produce a distorted image at one canvas size and an undistorted
image at a larger canvas (the undistorted canvas is expanded to avoid cropping
any source pixels). The AI\_colmap\_camera\_tracking pipeline generates this
pair as `*_transforms.json` (original) and
`undistort/transforms_undistorted.json` (expanded).

1. Set **Original JSON** to the original (distorted) `*_transforms.json`
2. Set **Undistorted JSON** to the expanded `transforms_undistorted.json`
3. Click **Load from JSON(s)**

The button loads both files in the correct order:
- K\_new intrinsics (focal, principal point, native canvas) ← undistorted JSON
- K\_orig intrinsics + distortion coefficients ← original JSON (overrides any zeroed values from the undistorted JSON)

In expand mode the node changes its output format dynamically:

| Mode | Input | Output |
| --- | --- | --- |
| Undistort | K\_orig canvas (distorted, smaller) | K\_new canvas (undistorted, larger) |
| Distort | K\_new canvas (undistorted, larger) | K\_orig canvas (distorted, smaller) |

Proportional scaling is applied automatically when the connected image
resolution differs from the calibration reference size.

> **Note:** Expand mode is perspective-only. Fisheye does not support two-canvas workflows.

Parameters can always be edited manually after loading.

---

## Project structure

```text
nuke-lens-distort-cv/
├── src/
│   └── LensDistort.cpp                    # Plugin source
├── triplets/
│   ├── x64-windows-static-md-v142.cmake   # vcpkg triplet for VS2019
│   └── x64-windows-static-md-v143.cmake   # vcpkg triplet for VS2022
├── CMakeLists.txt                         # Build system
├── CMakePresets.json                      # Presets: windows-vs2019, windows-vs2022, linux
├── vcpkg.json                             # vcpkg manifest (OpenCV calib3d + nlohmann-json)
└── README.md
```

---

## Notes

- **Remap maps are cached** — only rebuilt when knob values or image format
  changes. Scrubbing the timeline with fixed coefficients is fast.
- **Y-flip**: Nuke is bottom-up (row 0 = bottom); OpenCV is top-down.
  The plugin handles this transparently.
- **CRT**: Always compiled with `/MD` to match Nuke's runtime.
  Never use `/MT` — it causes heap corruption.
- **Compiler must match Nuke's**: Using the wrong Visual Studio version causes
  C++ ABI mismatches (exception handling, vtable layout) that crash Nuke at
  node creation. Use v142 for Nuke 15, v143 for Nuke 16/17.
- **OpenCV threading**: `cv::remap`, `cv::initUndistortRectifyMap`, and
  `cv::fisheye::initUndistortRectifyMap` use `parallel_for_` internally, which
  conflicts with Nuke 17's thread scheduler and causes crashes. All pixel-level
  work is done in plain single-threaded C++ loops. Only
  `cv::fisheye::estimateNewCameraMatrixForUndistortRectify` is called from
  OpenCV (pure matrix math, no threading).
- **Perspective distort** uses 10-iteration fixed-point (successive-
  approximation) iteration to invert the Brown-Conrady model — the same
  scheme as OpenCV's `undistortPoints`. **Perspective undistort** applies
  the forward model directly. The output camera matrix equals K (matching Python's `K.copy()`
  behaviour); the Alpha knob has no effect in perspective mode.
- **Fisheye distort** uses Newton iteration on θ to invert the equidistant
  model. **Fisheye undistort** applies the forward θ model directly.
  The output camera matrix is computed by
  `estimateNewCameraMatrixForUndistortRectify` with the configured Alpha.
- **DLL pinning** (Windows): The plugin pins itself in memory via
  `GetModuleHandleExW` with `GET_MODULE_HANDLE_EX_FLAG_PIN` to prevent
  Nuke from unloading the DLL and leaving stale `Op::Description` pointers.
