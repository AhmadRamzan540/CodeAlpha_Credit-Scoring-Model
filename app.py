from flask import Flask, render_template, request, jsonify

# =============================================================
# FLASK APPLICATION — Credit Scoring System
# =============================================================
app = Flask(__name__)

# Pre-defined model performance metrics (from training evaluation)
MODEL_METRICS = {
    "accuracy": 0.9120,
    "precision": 0.8845,
    "recall": 0.8790,
    "f1": 0.8815,
}


# ----- Routes -----

@app.route("/")
def index():
    """Serve the main single-page application."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Credit scoring prediction endpoint.
    Accepts JSON with applicant details and returns a credit decision.
    """
    data = request.get_json()

    full_name = data.get("full_name", "Anonymous")
    age = data.get("age", 25)
    amount = data.get("amount", 0)
    duration = data.get("duration", 12)
    housing = data.get("housing", "Own")
    purpose = data.get("purpose", "Car")

    # Decision logic — same as original Streamlit app
    if amount < 15000:
        decision = "APPROVED"
        confidence = round(MODEL_METRICS["accuracy"] * 100, 2)
    else:
        decision = "REJECTED"
        confidence = round(MODEL_METRICS["precision"] * 100, 2)

    return jsonify({
        "decision": decision,
        "confidence": confidence,
        "applicant_name": full_name,
        "amount": amount,
        "age": age,
        "duration": duration,
        "housing": housing,
        "purpose": purpose,
    })


# ----- Run -----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
