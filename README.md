# LensDistort — Nuke NDK Plugin

OpenCV-based lens distortion / undistortion node for Nuke 17+.

## Features

- **Undistort** — removes lens distortion (straighten footage)
- **Distort** — adds lens distortion (match a real lens)
- Full **Brown-Conrady** model: k1–k6, p1, p2 (rational polynomial)
- Configurable focal length and principal point
- Nearest / Bilinear / Bicubic filter modes
- Remap maps cached and only rebuilt when parameters change
- Self-contained: OpenCV statically linked into the DLL (no OpenCV DLLs on farm)

## Coefficients convention

Matches OpenCV / ShotCalibrate / most calibration tools:

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

---

## Build: Windows

OpenCV is managed by **vcpkg** — no manual compilation required.
The `x64-windows-static-md` triplet ensures OpenCV is statically linked
with `/MD` CRT, which matches Nuke's runtime.

### Prerequisites

| Tool | Version |
| --- | --- |
| Visual Studio | 2022 (v143) — must match Nuke 17's compiler |
| CMake | 3.20+ |
| Nuke | 17.0v1+ (for NDK headers + DDImage.lib) |
| vcpkg | latest |

### Step 1 — Install vcpkg (once)

```powershell
git clone https://github.com/microsoft/vcpkg.git C:/dev/vcpkg
C:/dev/vcpkg/bootstrap-vcpkg.bat
```

### Step 2 — Set VCPKG_ROOT (once)

```powershell
# Persist across sessions (recommended):
[System.Environment]::SetEnvironmentVariable("VCPKG_ROOT", "C:/dev/vcpkg", "User")

# Or for the current session only:
$env:VCPKG_ROOT = "C:/dev/vcpkg"
```

### Step 3 — Configure

`NUKE_ROOT` is required and must be passed explicitly.
vcpkg reads `vcpkg.json` and automatically downloads + compiles OpenCV.

```powershell
cd NukeLensDistort
cmake --preset windows -DNUKE_ROOT="C:/Program Files/Nuke17.0v1"
```

### Step 4 — Build

```powershell
cmake --build --preset windows
```

Output: `build/deploy/LensDistort.dll`

### Step 5 — Verify no OpenCV runtime deps

```powershell
# In VS 2022 Developer Command Prompt (x64):
dumpbin /dependents build\deploy\LensDistort.dll | findstr opencv
# Should return nothing — OpenCV is baked in
```

### Step 6 — Deploy

Copy `LensDistort.dll` to your Nuke plugin path:

```text
%USERPROFILE%/.nuke/plugins/LensDistort.dll
```

Or set `NUKE_PATH` to the folder containing the DLL.

---

## Build: Linux

OpenCV is also managed by vcpkg. The `x64-linux` triplet builds static,
PIC-enabled libraries.

### Prerequisites (Linux)

| Tool | Version |
| --- | --- |
| GCC or Clang | C++17-capable |
| CMake | 3.20+ |
| Ninja | any |
| Nuke | 15.0v1+ |
| vcpkg | latest |

### Steps

```bash
# Install vcpkg (once)
git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg
/opt/vcpkg/bootstrap-vcpkg.sh

# Set VCPKG_ROOT (add to ~/.bashrc for persistence)
export VCPKG_ROOT=/opt/vcpkg

# Configure — NUKE_ROOT is required
cd NukeLensDistort
cmake --preset linux -DNUKE_ROOT=/usr/local/Nuke17.0v1

# Build
cmake --build --preset linux

# Verify — should print nothing
ldd build/deploy/LensDistort.so | grep opencv
```

Output: `build/deploy/LensDistort.so`

---

## Usage in Nuke

1. Place `LensDistort.dll` / `LensDistort.so` in your `NUKE_PATH`
2. Create node: **Tab** → type `LensDistort`
3. Found under **Filter > LensDistort** in the node menu

### Typical workflow

**Undistort footage from a calibrated camera:**

```text
k1 = -0.28   (from calibration)
k2 =  0.07
k3 =  0.0
k4 =  0.0    (leave at 0 if calibration tool only outputs k1–k3)
k5 =  0.0
k6 =  0.0
p1 =  0.001
p2 = -0.0005
focal_x = 0  (auto)
focal_y = 0  (auto)
center_x = 0.5
center_y = 0.5
mode = Undistort
```

**Re-distort CG to match plate:**
Set the same coefficients, switch mode to **Distort**.

---

## Project structure

```text
NukeLensDistort/
├── src/
│   └── LensDistort.cpp     # Plugin source
├── CMakeLists.txt          # Build system
├── CMakePresets.json       # Platform presets (Windows / Linux)
├── vcpkg.json              # vcpkg manifest — declares OpenCV dependency
└── README.md
```

---

## Notes

- **Remap maps are cached** — only rebuilt when knob values or image format changes.
  Scrubbing timeline with fixed coefficients is fast.
- **Y-flip**: Nuke is bottom-up (row 0 = bottom); OpenCV is top-down.
  The plugin handles this transparently during copy in/out.
- **CRT**: Always compiled with `/MD` (dynamic CRT) to match Nuke's runtime.
  Never use `/MT` — it causes heap corruption.
- **Distort mode** uses the analytical forward Brown-Conrady rational model directly.
  Undistort mode uses `cv::initUndistortRectifyMap` with `getOptimalNewCameraMatrix`
  (alpha=0, no black borders). Both modes support the full 8-coefficient rational
  model (k1–k6, p1, p2); setting k4=k5=k6=0 is equivalent to a 5-coefficient
  polynomial calibration.
