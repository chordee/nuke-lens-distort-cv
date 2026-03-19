cmake_minimum_required(VERSION 3.20)
project(NukeLensDistort CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────────
# Paths — override these on the command line or in CMakePresets.json
# ─────────────────────────────────────────────────────────────────────────────

set(NUKE_ROOT "" CACHE PATH "Nuke installation root (required, e.g. -DNUKE_ROOT=\"C:/Program Files/Nuke17.0v1\")")
if(NOT NUKE_ROOT)
    message(FATAL_ERROR "NUKE_ROOT is not set.\n"
        "Pass it on the command line: cmake --preset windows -DNUKE_ROOT=<path>")
endif()

# OpenCV is managed by vcpkg (see vcpkg.json).
# Set VCPKG_ROOT env var and pass the toolchain file via CMakePresets.json;
# find_package below will resolve automatically.

# ─────────────────────────────────────────────────────────────────────────────
# Find packages
# ─────────────────────────────────────────────────────────────────────────────

# Nuke NDK — NukeConfig.cmake is shipped inside the Nuke install.
# It provides the add_nuke_plugin() helper and the Nuke::NDK target,
# which carry the correct include paths, DDImage.lib, NOMINMAX,
# _USE_MATH_DEFINES, and other NDK requirements automatically.
list(APPEND CMAKE_PREFIX_PATH "${NUKE_ROOT}")
find_package(Nuke REQUIRED)

# OpenCV — resolved via vcpkg toolchain
find_package(OpenCV REQUIRED COMPONENTS core calib3d)

# nlohmann/json — header-only, resolved via vcpkg toolchain
find_package(nlohmann_json CONFIG REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
# Plugin target
# ─────────────────────────────────────────────────────────────────────────────

# add_nuke_plugin() creates a MODULE library and handles:
#   - Nuke include dirs and DDImage linking
#   - PREFIX="" (no "lib" prefix)
#   - NOMINMAX and _USE_MATH_DEFINES on Windows
add_nuke_plugin(LensDistortCV src/LensDistort.cpp)

target_include_directories(LensDistortCV PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(LensDistortCV PRIVATE nlohmann_json::nlohmann_json)

# ── Windows-specific ──────────────────────────────────────────────────────────
if(WIN32)
    # CRT must match Nuke's runtime (/MD). add_nuke_plugin() does not set this.
    set_property(TARGET LensDistortCV PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

    target_compile_definitions(LensDistortCV PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )

    target_link_libraries(LensDistortCV PRIVATE
        ws2_32 comctl32 gdi32 user32  # required by OpenCV static build
        ${OpenCV_LIBS}
    )

    # MODULE libraries on Windows use LIBRARY_OUTPUT_DIRECTORY, not RUNTIME
    set_target_properties(LensDistortCV PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/deploy"
        SUFFIX ".dll"
    )
endif()

# ── Linux-specific ────────────────────────────────────────────────────────────
if(UNIX AND NOT APPLE)
    target_link_libraries(LensDistortCV PRIVATE
        "${NUKE_ROOT}/lib/libDDImage.so"
        -ldl -lm -lpthread -lz
    )

    set_target_properties(LensDistortCV PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/deploy"
        SUFFIX ".so"
    )

    # --whole-archive ensures no symbols are stripped from OpenCV static libs
    target_link_libraries(LensDistortCV PRIVATE
        -Wl,--whole-archive
        ${OpenCV_LIBS}
        -Wl,--no-whole-archive
    )
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────────────────────────────────────

install(TARGETS LensDistortCV
    RUNTIME DESTINATION "${NUKE_ROOT}/plugins"   # .dll on Windows
    LIBRARY DESTINATION "${NUKE_ROOT}/plugins"   # .so  on Linux
)

# ─────────────────────────────────────────────────────────────────────────────
# Build summary
# ─────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "── LensDistortCV build config ──────────────────────")
message(STATUS "  Nuke root    : ${NUKE_ROOT}")
message(STATUS "  OpenCV ver   : ${OpenCV_VERSION}")
message(STATUS "  OpenCV libs  : ${OpenCV_LIBS}")
message(STATUS "  Output       : ${CMAKE_BINARY_DIR}/deploy/LensDistortCV${CMAKE_SHARED_LIBRARY_SUFFIX}")
message(STATUS "──────────────────────────────────────────────────")
message(STATUS "")
