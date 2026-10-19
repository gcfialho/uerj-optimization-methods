from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import pytest

from nelder_mead.cli import main

matplotlib.use("Agg")


def test_cli_writes_history_and_gif(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--function", "booth", "--animate", "--output-dir", str(tmp_path)])
    assert "booth-2d" in capsys.readouterr().out

    rows = list(csv.DictReader((tmp_path / "booth-2d.csv").open(encoding="utf-8")))
    assert rows[0]["step"] == "initial"
    assert float(rows[-1]["f_best"]) < 1e-12
    assert int(rows[-1]["iteration"]) == len(rows) - 1

    gif = tmp_path / "booth-2d.gif"
    assert gif.read_bytes()[:6] == b"GIF89a"


def test_cli_3d_animation(tmp_path: Path) -> None:
    main(["-f", "shifted-sphere", "--max-iter", "15", "--animate", "--output-dir", str(tmp_path)])
    assert (tmp_path / "shifted-sphere-3d.gif").stat().st_size > 0


def test_cli_compare_scipy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["-f", "sphere", "-d", "4", "--compare-scipy", "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "sphere-4d" in out
    assert "scipy" in out


def test_cli_rejects_invalid_dimension() -> None:
    with pytest.raises(SystemExit):
        main(["-f", "himmelblau", "-d", "3"])
