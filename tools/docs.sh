#!/usr/bin/env bash
# Build documentation for digitalannealer

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

usage() {
    cat <<'USAGE'
Usage: tools/docs.sh [--preview]

Build project documentation.

Options:
  --preview   Build docs and serve docs/_build/html at http://127.0.0.1:8000
  -h, --help  Show this help message
USAGE
}

preview=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preview)
            if [[ "$preview" == true ]]; then
                echo "Error: --preview can only be provided once." >&2
                usage >&2
                exit 1
            fi
            preview=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'." >&2
            usage >&2
            exit 1
            ;;
    esac
done

echo "Building documentation..."
uv run --group docs sphinx-build -b html docs/source docs/_build/html -W --keep-going

echo "Documentation built at docs/_build/html"

if [[ "$preview" == true ]]; then
    echo "Preview Built Documentation"
    echo "Serving documentation at http://127.0.0.1:8000 (press Ctrl+C to stop)."
    uv run python -m http.server --bind 127.0.0.1 --directory docs/_build/html 8000
fi
