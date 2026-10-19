"""Flask application: serves the UI and a small JSON API."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template, request

from tank_optimizer.service import InvalidRequestError, run_comparison, run_optimization


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/optimize")
    def optimize() -> Any:
        """Run one method. JSON body: ``{D0, L0, method, tol, max_iter}``."""
        return _handle(run_optimization)

    @app.post("/compare")
    def compare() -> Any:
        """Run every method from the same start. JSON body: ``{D0, L0, tol, max_iter}``."""
        return _handle(run_comparison)

    return app


def _handle(action: Any) -> Any:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400
    try:
        return jsonify(action(data))
    except InvalidRequestError as exc:
        return jsonify({"error": str(exc)}), 400
