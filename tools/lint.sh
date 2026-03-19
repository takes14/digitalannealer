#!/usr/bin/env bash
# Lint and format script for digitalannealer

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Running Python linting with ruff..."
uv run --group lint ruff check digitalannealer/ tests/ examples/ --fix

echo "Running Python formatting with ruff..."
uv run --group lint ruff format digitalannealer/ tests/ examples/

echo "Running type checking with ty..."
uv run --group lint ty check digitalannealer/ tests/python/ examples/

echo "Running C++ formatting with clang-format..."
find src/ tests/cpp/ \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) -exec uv run --group lint clang-format -i {} +

echo "Done!"
