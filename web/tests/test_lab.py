"""The browser lab runs web/lab.py under Pyodide; check its payloads here, in CPython."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from nelder_mead import PROBLEMS, minimize

LAB = Path(__file__).resolve().parents[1] / "lab.py"
spec = importlib.util.spec_from_file_location("lab", LAB)
assert spec is not None
assert spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_nm_run_matches_the_package_and_is_json() -> None:
    out = lab.nm_run("rosenbrock")
    demo = PROBLEMS["rosenbrock-2d"]
    ref = minimize(
        demo.function.func, initial_simplex=demo.initial_simplex, max_iter=400, max_fev=1600
    )
    assert out["nit"] == ref.nit
    assert out["nfev"] == ref.nfev
    np.testing.assert_allclose(out["x"], ref.x)
    assert len(out["simplices"]) == len(out["steps"]) == out["nit"] + 1
    assert sum(out["counts"].values()) == out["nit"]
    json.dumps(out)


def test_nm_run_from_a_clicked_start() -> None:
    out = lab.nm_run("himmelblau", start=[3.0, 3.0])
    assert out["converged"]
    assert out["distance"] < 1e-3


def test_nm_run_rejects_bad_coefficients() -> None:
    with pytest.raises(ValueError, match="contraction"):
        lab.nm_run("booth", coefficients={"contraction": 1.5})


def test_nm_functions_grids_are_json() -> None:
    funcs = lab.nm_functions()
    assert set(funcs) == {"rosenbrock", "booth", "himmelblau", "sphere"}
    assert len(funcs["booth"]["z"]) == lab.GRID
    json.dumps(funcs)


def test_tank_compare_payload_is_json() -> None:
    out = lab.tank_compare({"D0": 0.45, "L0": 0.55, "tol": 1e-6, "max_iter": 200})
    assert set(out["results"]) == {"SD", "Newton", "DFP"}
    assert out["results"]["Newton"]["feasible"]
    json.dumps(out)


def test_call_is_the_json_bridge() -> None:
    ok = json.loads(lab.call("nm_run", json.dumps({"function": "booth"})))
    assert ok["converged"]
    bad = json.loads(lab.call("nm_run", json.dumps({"function": "nope"})))
    assert "unknown function" in bad["error"]
    tank = json.loads(lab.call("tank_run", json.dumps({"D0": 0.5, "L0": 0.5, "method": "Newton"})))
    assert tank["feasible"]
