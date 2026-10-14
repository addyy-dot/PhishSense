
from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import urlsplit, parse_qs
import pandas as pd
import numpy as np
import joblib
import re
import math
import ipaddress
import os
from collections import Counter

app = Flask(__name__)

# Allow requests from your frontend.
# For production, restrict this to your actual frontend origin.
CORS(app)

# Load the improved URL-only models, scaler, and feature order.
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

# Load scaler for Logistic Regression (if available).
scaler_path = os.path.join(BASE_DIR, "url_only_scaler.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# ---------------------------------------------------------------------------
# Trusted domain whitelist
# These well-known domains are given a legitimacy boost to prevent
# obvious false positives. The model still runs; the whitelist only
# adjusts the final probability to avoid embarrassing misclassifications.
# ---------------------------------------------------------------------------
TRUSTED_DOMAINS = {
    # Search & social
    "google.com", "youtube.com", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "linkedin.com", "reddit.com",
    "pinterest.com", "tiktok.com", "snapchat.com", "whatsapp.com",
    "telegram.org", "discord.com", "twitch.tv",

    # Tech
    "microsoft.com", "apple.com", "amazon.com", "github.com",
    "gitlab.com", "stackoverflow.com", "mozilla.org",
    "wikipedia.org", "wikimedia.org",

    # Email & productivity
    "outlook.com", "live.com", "office.com", "zoom.us",
    "slack.com", "notion.so", "trello.com",

    # Finance
    "paypal.com", "stripe.com", "chase.com", "bankofamerica.com",
    "wellsfargo.com", "visa.com", "mastercard.com",

    # Shopping
    "ebay.com", "walmart.com", "target.com", "bestbuy.com",
    "shopify.com", "etsy.com",

    # Entertainment & media
    "netflix.com", "spotify.com", "hulu.com", "disneyplus.com",
    "nytimes.com", "bbc.com", "cnn.com", "reuters.com",

    # Cloud / developer platforms
    "aws.amazon.com", "cloud.google.com", "azure.microsoft.com",
    "vercel.com", "netlify.com", "heroku.com",
    "cloudflare.com", "digitalocean.com",

    # Education
    "coursera.org", "edx.org", "khanacademy.org",

    # Government
    "irs.gov", "usa.gov", "gov.uk",
}


def is_trusted_domain(hostname):
    """
    Check if the hostname belongs to a trusted domain.
    Handles subdomains like www.google.com, mail.google.com.
    """
    hostname = hostname.lower().strip()
    # Direct match
    if hostname in TRUSTED_DOMAINS:
        return True
    # Subdomain match (e.g., www.google.com → google.com)
    parts = hostname.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in TRUSTED_DOMAINS:
            return True
    return False


# ---------------------------------------------------------------------------
# Feature extraction (same as retrain.py)
# ---------------------------------------------------------------------------

COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "co", "io", "us", "uk", "ca", "au", "de", "fr",
    "in", "jp", "cn", "br", "ru", "it", "es", "nl",
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "secure", "account",
    "update", "confirm", "banking", "password", "credential",
    "suspend", "alert", "urgent", "paypal", "apple", "microsoft",
    "amazon", "netflix", "facebook", "google", "support", "help",
    "recover", "restore", "unlock", "validate", "authenticate",
    "webscr", "cmd", "wallet", "billing", "invoice",
]

SUSPICIOUS_HOSTING = [
    "000webhostapp.com", "weebly.com", "wixsite.com",
    "blogspot.com", "wordpress.com", "herokuapp.com",
    "firebaseapp.com", "web.app", "netlify.app",
    "vercel.app", "pages.dev", "glitch.me",
    "replit.co", "github.io",
]


def shannon_entropy(text):
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    counter = Counter(text)
    length = len(text)
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in counter.values()
    )
    return round(entropy, 6)


def extract_url_features(value):
    """Extract the improved feature set used during training."""

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

    # --- Basic URL features ---
    url_length = len(raw)
    n = max(url_length, 1)

    # --- Domain features ---
    domain_length = len(host)

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    parts = host.split(".")
    tld = parts[-1] if len(parts) > 1 else ""
    tld_length = len(tld)
    num_subdomains = max(0, len(parts) - 2) if not is_ip else 0
    is_common_tld = int(tld in COMMON_TLDS)

    domain_no_tld = ".".join(parts[:-1]) if len(parts) > 1 else host

    # --- Path features ---
    path = parsed.path or ""
    path_length = len(path)
    path_depth = path.strip("/").count("/") + 1 if path.strip("/") else 0

    # --- Query features ---
    query = parsed.query or ""
    query_length = len(query)
    query_params = parse_qs(query)
    num_query_params = len(query_params)

    # --- Character analysis ---
    letters = sum(c.isalpha() for c in raw)
    digits = sum(c.isdigit() for c in raw)

    letter_ratio = letters / n
    digit_ratio = digits / n

    num_dots = raw.count(".")
    num_hyphens = raw.count("-")
    num_slashes = raw.count("/")
    num_at = raw.count("@")
    num_percent = raw.count("%")
    num_equals = raw.count("=")
    num_ampersand = raw.count("&")

    obfuscated = num_at + num_percent
    has_obfuscation = int(obfuscated > 0)

    other_special = sum(
        not c.isalnum() and c not in "/:?&=.#-_~@%+,"
        for c in raw
    )
    special_ratio = other_special / n

    is_https = int(parsed.scheme.lower() == "https")

    url_entropy = shannon_entropy(raw)
    domain_entropy = shannon_entropy(host)

    url_lower = raw.lower()

    suspicious_keyword_count = sum(
        1 for kw in SUSPICIOUS_KEYWORDS
        if kw in url_lower
    )

    domain_has_suspicious_keyword = int(any(
        kw in domain_no_tld.lower()
        for kw in ["login", "signin", "verify", "secure", "account",
                    "update", "confirm", "banking", "password"]
    ))

    hex_char_count = len(re.findall(r"%[0-9a-fA-F]{2}", raw))

    has_long_subdomain = int(any(
        len(p) > 20 for p in parts[:-2]
    )) if len(parts) > 2 else 0

    uses_suspicious_hosting = int(any(
        host.endswith(h) for h in SUSPICIOUS_HOSTING
    ))

    return {
        "URLLength": url_length,
        "PathLength": path_length,
        "PathDepth": path_depth,
        "QueryLength": query_length,
        "NumQueryParams": num_query_params,
        "DomainLength": domain_length,
        "IsDomainIP": is_ip,
        "TLDLength": tld_length,
        "NumSubdomains": num_subdomains,
        "IsCommonTLD": is_common_tld,
        "DomainEntropy": domain_entropy,
        "NumLetters": letters,
        "LetterRatio": letter_ratio,
        "NumDigits": digits,
        "DigitRatio": digit_ratio,
        "NumDots": num_dots,
        "NumHyphens": num_hyphens,
        "NumSlashes": num_slashes,
        "HasObfuscation": has_obfuscation,
        "NumObfuscatedChars": obfuscated,
        "NumOtherSpecialChars": other_special,
        "SpecialCharRatio": special_ratio,
        "HexCharCount": hex_char_count,
        "NumEquals": num_equals,
        "NumAmpersand": num_ampersand,
        "IsHTTPS": is_https,
        "URLEntropy": url_entropy,
        "SuspiciousKeywordCount": suspicious_keyword_count,
        "DomainHasSuspiciousKeyword": domain_has_suspicious_keyword,
        "HasLongSubdomain": has_long_subdomain,
        "UsesSuspiciousHosting": uses_suspicious_hosting,
    }


def predict_with_model(model, row, use_scaler=False):
    """Return prediction and phishing probability."""

    input_data = row
    if use_scaler and scaler is not None:
        input_data = pd.DataFrame(
            scaler.transform(row),
            columns=row.columns
        )

    prediction = int(model.predict(input_data)[0])

    classes = list(model.classes_)
    phishing_index = classes.index(1)

    probabilities = model.predict_proba(input_data)[0]
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

        lr_result = predict_with_model(lr_model, row, use_scaler=True)
        xgb_result = predict_with_model(xgb_model, row, use_scaler=False)

        # --- Trusted domain override ---
        normalized = url.strip()
        if not re.match(r"^https?://", normalized, re.IGNORECASE):
            normalized = "https://" + normalized
        try:
            hostname = urlsplit(normalized).hostname or ""
        except ValueError:
            hostname = ""

        trusted = is_trusted_domain(hostname)

        if trusted:
            # Clamp phishing probability down for trusted domains.
            # The model may still see some URL features as suspicious
            # (e.g., short URL with few features), but we know this
            # domain is legitimate.
            lr_result["phishing_probability"] = min(
                lr_result["phishing_probability"], 0.05
            )
            lr_result["prediction"] = "Legitimate"

            xgb_result["phishing_probability"] = min(
                xgb_result["phishing_probability"], 0.05
            )
            xgb_result["prediction"] = "Legitimate"

        return jsonify({
            "url": url.strip(),
            "feature_count": len(feature_columns),
            "logistic_regression": lr_result,
            "xgboost": xgb_result,
            "trusted_domain": trusted,
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
