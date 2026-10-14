
from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import urlsplit
import pandas as pd
import joblib
import re
import ipaddress
import os

app = Flask(__name__)

# Allow requests from your frontend.
# For production, restrict this to your actual frontend origin.
CORS(app)

# Load the NEW URL-only models and their feature order.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

lr_model = joblib.load(
    os.path.join(BASE_DIR, "url_only_logistic_regression.pkl")
)

xgb_model = joblib.load(
    os.path.join(BASE_DIR, "url_only_xgboost.pkl")
)

feature_columns = joblib.load(
    os.path.join(BASE_DIR, "url_only_feature_columns.pkl")
)


def extract_url_features(value):
    """Extract the same 18 URL-only features used during training."""

    raw = str(value).strip()

    if not raw:
        raise ValueError("Please enter a URL.")

    # Add HTTPS if the user did not enter a scheme.
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        raw = "https://" + raw

    try:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
    except ValueError:
        raise ValueError("Invalid URL format.")

    if not host:
        raise ValueError("The URL must contain a hostname.")

    # Check whether the hostname is an IP address.
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    parts = host.split(".")
    tld_length = len(parts[-1]) if len(parts) > 1 else 0
    subdomains = max(0, len(parts) - 2) if not is_ip else 0

    n = max(len(raw), 1)

    letters = sum(c.isalpha() for c in raw)
    digits = sum(c.isdigit() for c in raw)
    obfuscated = sum(c in "%@" for c in raw)

    other_special = sum(
        not c.isalnum() and c not in "/:?&=.#-_"
        for c in raw
    )

    return {
        "URLLength": len(raw),
        "DomainLength": len(host),
        "IsDomainIP": is_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": subdomains,
        "HasObfuscation": int(obfuscated > 0),
        "NoOfObfuscatedChar": obfuscated,
        "ObfuscationRatio": obfuscated / n,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / n,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / n,
        "NoOfEqualsInURL": raw.count("="),
        "NoOfQMarkInURL": raw.count("?"),
        "NoOfAmpersandInURL": raw.count("&"),
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": other_special / n,
        "IsHTTPS": int(parsed.scheme.lower() == "https")
    }


def predict_with_model(model, row):
    """Return prediction and phishing probability."""

    prediction = int(model.predict(row)[0])

    classes = list(model.classes_)
    phishing_index = classes.index(1)

    probabilities = model.predict_proba(row)[0]
    phishing_probability = float(probabilities[phishing_index])

    return {
        "prediction": (
            "Phishing" if prediction == 1 else "Legitimate"
        ),
        "phishing_probability": round(phishing_probability, 4)
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "PhishSense URL-only API is running!",
        "endpoint": "/predict-url",
        "method": "POST"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": True,
        "feature_count": len(feature_columns)
    })


@app.route("/predict-url", methods=["POST"])
def predict_url():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object containing a URL."
        }), 400

    url = data.get("url")

    if not isinstance(url, str) or not url.strip():
        return jsonify({
            "error": "Please provide a non-empty URL."
        }), 400

    if len(url) > 2048:
        return jsonify({
            "error": "URL is too long. Maximum length is 2048 characters."
        }), 400

    try:
        features = extract_url_features(url)

        # Ensure feature order matches training.
        row = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns
        )

        lr_result = predict_with_model(lr_model, row)
        xgb_result = predict_with_model(xgb_model, row)

        return jsonify({
            "url": url.strip(),
            "feature_count": len(feature_columns),
            "logistic_regression": lr_result,
            "xgboost": xgb_result,
            "notice": (
                "Predictions use URL text only. "
                "Webpage content is not inspected."
            )
        })

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the backend logs."
        }), 500


if __name__ == "__main__":
    # Local development only. Render uses Gunicorn.
    app.run(host="127.0.0.1", port=5000, debug=False)
