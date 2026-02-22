import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ── Load model artefacts once at startup ─────────────────────────────────────
model = pickle.load(open("models/car_price_model.pkl", "rb"))
model_columns = pickle.load(open("models/model_columns.pkl", "rb"))

# ── Helpers ───────────────────────────────────────────────────────────────────
def simplify_transmission(x: str) -> str:
    x = str(x).lower()
    if "manual" in x or "m/t" in x:
        return "Manual"
    elif "cvt" in x:
        return "CVT"
    elif "dual" in x or "dct" in x:
        return "Dual-Clutch"
    elif "auto" in x or "a/t" in x:
        return "Automatic"
    return "Other"


def preprocess(form) -> pd.DataFrame:
    """Turn raw form values into a model-ready DataFrame."""
    transmission_simplified = simplify_transmission(form["transmission"])

    raw = pd.DataFrame([{
        "model_year":  int(form["model_year"]),
        "milage":      int(form["milage"]),
        "fuel_type":   form["fuel_type"],
        "accident":    form["accident"],
        "clean_title": form["clean_title"],
        "transmission": transmission_simplified,
        "brand":       form["brand"],
    }])

    # One-hot encode exactly as in training
    cats = ["fuel_type", "accident", "clean_title", "transmission", "brand"]
    encoded = pd.get_dummies(raw, columns=cats, dtype=int)

    # Align to training schema (fills missing dummies with 0)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    return encoded


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            input_df = preprocess(request.form)
            log_price = model.predict(input_df)[0]
            prediction = round(float(np.exp(log_price)), 2)
        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
