
# H5Z-MANS: HDF5 Compression Filter Plugin For MANS


## Features

* **Dynamic Loading:** Built as a shared library (`.so` / `.dll`), allowing usage without recompiling HDF5 applications.
* **Dual Backend:**
* **CPU:** Multi-threaded acceleration using OpenMP (AVX512 optimizations).
* **GPU:** (Experimental) NVIDIA CUDA acceleration.


* **Configurable:** Runtime tuning of compression parameters (thresholds, data types) via filter parameters (`cd_values`).

---

## Build Instructions

This plugin is built as part of the main MANS project. Ensure the `BUILD_HDF5_PLUGIN` option is enabled.

```bash
mkdir build && cd build

# Configure (Ensure BUILD_HDF5_PLUGIN is ON)
cmake .. -DBUILD_HDF5_PLUGIN=ON 

# Build
make -j

```

### Build Artifacts

After a successful build, the artifacts will be located at:

* **Plugin Library:** `build/bin/plugins/libH5Z-MANS.so`
* **Test Utility:** `build/bin/h5z-mans/H5Z-MANS_test`

---

## Usage

### Environment Setup (Crucial)

HDF5 requires the `HDF5_PLUGIN_PATH` environment variable to locate external filters. You must point this to the directory containing `libH5Z-MANS.so`.

```bash
# Assuming you are in the project root/build directory
export HDF5_PLUGIN_PATH=$(pwd)/bin/plugins

```

## Testing & Verification

A dedicated test tool `H5Z-MANS_test` is provided to verify compression integrity (Bit-Exact check) and measure compression ratio.

### 1. Create a Configuration File

see [example.conf](example.conf)

### 2. Run the Test

```bash
# 1. Export Plugin Path
export HDF5_PLUGIN_PATH=<path of build>/bin/plugins

# 2. Run Test
# Usage: ./H5Z-MANS_test <config_file> <output_h5_file>
./bin/h5z-mans/H5Z-MANS_test example.conf output.h5

```
