"""Launch the Turbo Dodge AI game.

Run this file from an IDE, or from the repository root with:

    .venv\\Scripts\\python.exe "Day 5\\project.py"
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    from turbo_dodge_ai.app import run_app

    return run_app(project_dir)


if __name__ == "__main__":
    raise SystemExit(main())
