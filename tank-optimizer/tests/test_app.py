from __future__ import annotations

import pytest
from flask.testing import FlaskClient

from tank_optimizer.app import create_app
from tank_optimizer.cli import main


@pytest.fixture
def client() -> FlaskClient:
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_index(client: FlaskClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"Cylindrical Tank" in response.data
    assert client.get("/static/js/main.js").status_code == 200


def test_optimize(client: FlaskClient) -> None:
    response = client.post("/optimize", json={"D0": 0.5, "L0": 0.5, "method": "DFP"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["feasible"] is True
    assert data["iterations"] == len(data["history"]) - 1 == len(data["errors"]) - 1
    assert data["num_eval"] > data["iterations"]
    assert len(data["contour"]["cost"]) == len(data["contour"]["l"])
    assert data["contour"]["limits"]["v_min"] == pytest.approx(0.72)


def test_compare(client: FlaskClient) -> None:
    response = client.post("/compare", json={"D0": 0.8, "L0": 1.2, "tol": "1e-6"})
    assert response.status_code == 200
    results = response.get_json()["results"]
    assert set(results) == {"SD", "Newton", "DFP"}
    assert all(r["feasible"] for r in results.values())


@pytest.mark.parametrize(
    ("route", "payload"),
    [
        ("/optimize", {"D0": 0.5, "L0": 0.5, "method": "BFGS"}),
        ("/optimize", {"D0": "abc", "L0": 0.5}),
        ("/compare", {"L0": 0.5}),
        ("/compare", {"D0": -1, "L0": 0.5}),
        ("/compare", {"D0": 0.5, "L0": 0.5, "max_iter": 0}),
    ],
)
def test_invalid_requests(client: FlaskClient, route: str, payload: dict) -> None:
    response = client.post(route, json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_json_body(client: FlaskClient) -> None:
    response = client.post("/compare", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_cli_compare(capsys: pytest.CaptureFixture[str]) -> None:
    main(["compare", "--d0", "0.5", "--l0", "0.5"])
    out = capsys.readouterr().out
    for method in ("SD", "Newton", "DFP"):
        assert method in out
