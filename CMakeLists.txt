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

# Nuke NDK — uses NukeConfig.cmake shipped inside the Nuke install
list(APPEND CMAKE_PREFIX_PATH "${NUKE_ROOT}")
find_package(Nuke REQUIRED)          # provides Nuke::NDK target

# OpenCV — resolved via vcpkg toolchain
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)

# ─────────────────────────────────────────────────────────────────────────────
# Plugin target
# ─────────────────────────────────────────────────────────────────────────────

add_library(LensDistort SHARED src/LensDistort.cpp)

# Nuke expects the plugin filename to match the class name exactly
set_target_properties(LensDistort PROPERTIES
    PREFIX  ""               # no 'lib' prefix on Linux either
    OUTPUT_NAME "LensDistort"
)

target_include_directories(LensDistort PRIVATE
    "${NUKE_ROOT}/include"
    ${OpenCV_INCLUDE_DIRS}
)

# ── Windows-specific ──────────────────────────────────────────────────────────
if(WIN32)
    # Must match Nuke's CRT: /MD (dynamic multithreaded release)
    # NEVER use /MT — CRT mismatch causes heap corruption
    set_property(TARGET LensDistort PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

    # Suppress warnings about std::memcpy with MSVC
    target_compile_definitions(LensDistort PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX           # prevent Windows.h min/max macros
    )

    # Nuke NDK .lib for linking on Windows
    target_link_libraries(LensDistort PRIVATE
        "${NUKE_ROOT}/lib/DDImage.lib"
    )

    # Windows system libs that OpenCV static build requires
    target_link_libraries(LensDistort PRIVATE
        ws2_32 comctl32 gdi32 user32
    )

    # Output directly to a deploy folder for convenience
    set_target_properties(LensDistort PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/deploy"
        SUFFIX ".dll"
    )
endif()

# ── Linux-specific ────────────────────────────────────────────────────────────
if(UNIX AND NOT APPLE)
    target_link_libraries(LensDistort PRIVATE
        "${NUKE_ROOT}/lib/libDDImage.so"
        -ldl -lm -lpthread -lz
    )

    set_target_properties(LensDistort PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/deploy"
        SUFFIX ".so"
    )

    # When OpenCV is static, its .a files must be position-independent
    # (built with -DCMAKE_POSITION_INDEPENDENT_CODE=ON).
    # --whole-archive must directly bracket the OpenCV .a files in the link
    # command; target_link_options cannot guarantee that ordering, so pass the
    # flags through target_link_libraries to ensure correct interleaving.
    target_link_libraries(LensDistort PRIVATE
        -Wl,--whole-archive
        ${OpenCV_LIBS}
        -Wl,--no-whole-archive
    )
endif()

# ── OpenCV libs (Windows) ─────────────────────────────────────────────────────
# On Linux, OpenCV is already linked above with --whole-archive wrapping.
if(WIN32)
    target_link_libraries(LensDistort PRIVATE ${OpenCV_LIBS})
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────────────────────────────────────

install(TARGETS LensDistort
    RUNTIME DESTINATION "${NUKE_ROOT}/plugins"   # .dll on Windows
    LIBRARY DESTINATION "${NUKE_ROOT}/plugins"   # .so  on Linux
)

# ─────────────────────────────────────────────────────────────────────────────
# Build summary
# ─────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "── LensDistort build config ──────────────────────")
message(STATUS "  Nuke root    : ${NUKE_ROOT}")
message(STATUS "  OpenCV ver   : ${OpenCV_VERSION}")
message(STATUS "  OpenCV libs  : ${OpenCV_LIBS}")
message(STATUS "  Output       : ${CMAKE_BINARY_DIR}/deploy/LensDistort${CMAKE_SHARED_LIBRARY_SUFFIX}")
message(STATUS "──────────────────────────────────────────────────")
message(STATUS "")
