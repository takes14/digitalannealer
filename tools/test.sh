#!/usr/bin/env bash
# Test script for digitalannealer

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BUILD_DIR="build"
readonly BUILD_TYPE="Debug"

cd "${REPO_ROOT}"

banner() {
    local message="$1"
    printf '\n============================================================\n'
    printf '%s\n' "${message}"
    printf '============================================================\n'
}

banner "Step 1/3: Building the project with CMake"
uv run --group test cmake -B "${BUILD_DIR}" -S . -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DBUILD_TESTS=ON
uv run --group test cmake --build "${BUILD_DIR}" -j"$(nproc)"

banner "Step 2/3: Running C++ tests (tests/cpp)"
if ! uv run --group test ctest --test-dir "${BUILD_DIR}" --output-on-failure; then
    echo "C++ tests failed. Stopping before Python tests."
    exit 1
fi

banner "Step 3/3: Running Python tests (tests/python)"
uv run --group test pytest tests/python

banner "All tests completed successfully."
