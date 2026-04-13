"""Run the Python smoke tests against an in-tree CMake build."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    extension_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None

    sys.path.insert(0, str(repo_root))
    if extension_dir is not None:
        sys.path.insert(0, str(extension_dir))

    from test_imcts import main as run_smoke_tests

    run_smoke_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
