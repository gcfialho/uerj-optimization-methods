"""Assemble the static lab in site/: the web page plus unmodified copies of the packages.

The page loads the Python files into Pyodide, so the browser runs the same code that the
tests cover. Usage: ``python scripts/build_site.py`` (no dependencies).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site"

PYTHON = {
    "nelder_mead": ROOT / "nelder-mead/src/nelder_mead",
    "tank_optimizer": ROOT / "tank-optimizer/src/tank_optimizer",
}
# Browser-irrelevant modules (plotting, CLIs, the Flask app) are left out.
SKIP = {"visualize.py", "cli.py", "app.py"}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(
        ROOT / "web", OUT, ignore=shutil.ignore_patterns("tests", "__pycache__", "lab.py")
    )
    files = []
    for package, src in PYTHON.items():
        for path in sorted(src.glob("*.py")):
            if path.name in SKIP:
                continue
            target = OUT / "py" / package / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files.append(f"{package}/{path.name}")
    shutil.copy2(ROOT / "web/lab.py", OUT / "py/lab.py")
    files.append("lab.py")
    (OUT / "py/manifest.json").write_text(json.dumps(files, indent=2) + "\n")
    print(f"site/ ready: {len(files)} Python files")


if __name__ == "__main__":
    main()
