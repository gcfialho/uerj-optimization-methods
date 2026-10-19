from flask import Flask, render_template, request, jsonify

from controllers.tank_controller import run_optimization, run_comparison

app = Flask(__name__)

@app.route("/")
def index():
    """Renders the main page of the application."""
    return render_template("index.html")

@app.route("/optimize", methods=["POST"])
def optimize():
    """
    Handles the optimization request for a single method.
    Expects a JSON payload with optimization parameters.
    """
    data = request.json
    if data is None:
        return jsonify({"error": "Invalid JSON payload"}), 400
    result = run_optimization(data)
    return jsonify(result)

@app.route("/compare", methods=["POST"])
def compare():
    """
    Handles the comparison request for all optimization methods.
    Expects a JSON payload with optimization parameters.
    """
    data = request.json
    if data is None:
        return jsonify({"error": "Invalid JSON payload"}), 400
    result = run_comparison(data)
    return jsonify(result)

if __name__ == "__main__":
    # Runs the Flask application in debug mode.
    app.run(debug=True)