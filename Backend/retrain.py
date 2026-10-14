"""
PhishSense — Improved Model Training Script
=============================================
This script retrains both Logistic Regression and XGBoost models with:
1. Correct label handling (dataset label 0 = phishing, 1 = legitimate)
2. Better feature engineering (30+ features including entropy, suspicious patterns)
3. Consistent feature extraction between training and inference
4. A trusted domain whitelist layer

Run this script from the Backend/ directory:
    python retrain.py
"""

import os
import re
import math
import ipaddress
import warnings
from collections import Counter
from urllib.parse import urlsplit, parse_qs

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Feature Extraction (shared between training and inference)
# ---------------------------------------------------------------------------

# Common legitimate TLDs
COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "co", "io", "us", "uk", "ca", "au", "de", "fr",
    "in", "jp", "cn", "br", "ru", "it", "es", "nl",
}

# Suspicious keywords often found in phishing URLs
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "secure", "account",
    "update", "confirm", "banking", "password", "credential",
    "suspend", "alert", "urgent", "paypal", "apple", "microsoft",
    "amazon", "netflix", "facebook", "google", "support", "help",
    "recover", "restore", "unlock", "validate", "authenticate",
    "webscr", "cmd", "wallet", "billing", "invoice",
]

# Free hosting / URL shortener domains (often abused by phishers)
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


def extract_features(url_string):
    """
    Extract 30 features from a URL string.
    This function is used both during training AND inference.
    """
    raw = str(url_string).strip()

    # Add scheme if missing
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

    # Check if domain is an IP address
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

    # Domain without TLD
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

    # Count specific characters
    num_dots = raw.count(".")
    num_hyphens = raw.count("-")
    num_underscores = raw.count("_")
    num_slashes = raw.count("/")
    num_at = raw.count("@")
    num_tilde = raw.count("~")
    num_percent = raw.count("%")
    num_equals = raw.count("=")
    num_qmark = raw.count("?")
    num_ampersand = raw.count("&")

    # Obfuscation characters (@ and %)
    obfuscated = num_at + num_percent
    has_obfuscation = int(obfuscated > 0)

    # Other special characters (not part of normal URL syntax)
    other_special = sum(
        not c.isalnum() and c not in "/:?&=.#-_~@%+,"
        for c in raw
    )
    special_ratio = other_special / n

    # --- Protocol features ---
    is_https = int(parsed.scheme.lower() == "https")

    # --- Entropy ---
    url_entropy = shannon_entropy(raw)
    domain_entropy = shannon_entropy(host)

    # --- Suspicious pattern detection ---
    url_lower = raw.lower()
    path_lower = path.lower()

    # Count suspicious keywords in the URL
    suspicious_keyword_count = sum(
        1 for kw in SUSPICIOUS_KEYWORDS
        if kw in url_lower
    )

    # Check if domain contains suspicious keywords (NOT the brand itself,
    # but when it appears in a subdomain of an unrelated domain)
    domain_has_suspicious_keyword = int(any(
        kw in domain_no_tld.lower()
        for kw in ["login", "signin", "verify", "secure", "account",
                    "update", "confirm", "banking", "password"]
    ))

    # Check for IP-like patterns in path
    has_ip_in_path = int(bool(re.search(
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", path
    )))

    # Check for hex-encoded characters
    hex_char_count = len(re.findall(r"%[0-9a-fA-F]{2}", raw))

    # Check for data URI or javascript-like patterns
    has_suspicious_scheme = int(bool(re.search(
        r"(data:|javascript:|vbscript:)", url_lower
    )))

    # Long random-looking subdomain check
    has_long_subdomain = int(any(
        len(p) > 20 for p in parts[:-2]
    )) if len(parts) > 2 else 0

    # Uses a suspicious free hosting platform
    uses_suspicious_hosting = int(any(
        host.endswith(h) for h in SUSPICIOUS_HOSTING
    ))

    return {
        # Basic URL
        "URLLength": url_length,
        "PathLength": path_length,
        "PathDepth": path_depth,
        "QueryLength": query_length,
        "NumQueryParams": num_query_params,

        # Domain
        "DomainLength": domain_length,
        "IsDomainIP": is_ip,
        "TLDLength": tld_length,
        "NumSubdomains": num_subdomains,
        "IsCommonTLD": is_common_tld,
        "DomainEntropy": domain_entropy,

        # Character counts
        "NumLetters": letters,
        "LetterRatio": letter_ratio,
        "NumDigits": digits,
        "DigitRatio": digit_ratio,
        "NumDots": num_dots,
        "NumHyphens": num_hyphens,
        "NumSlashes": num_slashes,

        # Obfuscation
        "HasObfuscation": has_obfuscation,
        "NumObfuscatedChars": obfuscated,
        "NumOtherSpecialChars": other_special,
        "SpecialCharRatio": special_ratio,
        "HexCharCount": hex_char_count,

        # Query
        "NumEquals": num_equals,
        "NumAmpersand": num_ampersand,

        # Protocol
        "IsHTTPS": is_https,

        # Entropy
        "URLEntropy": url_entropy,

        # Suspicious patterns
        "SuspiciousKeywordCount": suspicious_keyword_count,
        "DomainHasSuspiciousKeyword": domain_has_suspicious_keyword,
        "HasLongSubdomain": has_long_subdomain,
        "UsesSuspiciousHosting": uses_suspicious_hosting,
    }


FEATURE_NAMES = list(extract_features("https://example.com").keys())


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def main():
    # Check for dataset
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "PhiUSIIL_Phishing_URL_Dataset.csv"
    )

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please place PhiUSIIL_Phishing_URL_Dataset.csv in the Backend/ folder.")
        return

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # Clean data
    data = df.dropna(subset=["URL", "label"]).copy()
    data = data.drop_duplicates(subset=["URL"]).copy()
    print(f"  After cleaning: {len(data)} rows")

    # Original dataset labels: 0 = Phishing, 1 = Legitimate
    # Our model: 1 = Phishing, 0 = Legitimate
    print("\nOriginal label distribution:")
    print(f"  label=0 (Phishing): {(data['label'] == 0).sum()}")
    print(f"  label=1 (Legitimate): {(data['label'] == 1).sum()}")

    y = (data["label"] == 0).astype(int)
    print(f"\nMapped labels: 1=Phishing ({y.sum()}), 0=Legitimate ({(y == 0).sum()})")

    # Extract features from URLs
    print("\nExtracting features from URLs (this may take a few minutes)...")
    features_list = []
    errors = 0
    for i, url in enumerate(data["URL"]):
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i}/{len(data)} URLs...")
        try:
            features_list.append(extract_features(url))
        except Exception:
            features_list.append(extract_features("https://error.invalid"))
            errors += 1

    X = pd.DataFrame(features_list, columns=FEATURE_NAMES)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"  Feature matrix: {X.shape}")
    if errors > 0:
        print(f"  Errors during extraction: {errors}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\n  Training: {len(X_train)}, Testing: {len(X_test)}")

    # --- Scale features for Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    lr_model.fit(X_train_scaled, y_train)

    lr_preds = lr_model.predict(X_test_scaled)
    print(f"  Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print(classification_report(
        y_test, lr_preds,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))

    # --- Train XGBoost ---
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)

    xgb_preds = xgb_model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
    print(classification_report(
        y_test, xgb_preds,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))

    # --- Verify with known URLs ---
    print("\n=== Verification with known URLs ===")
    test_urls = [
        "https://google.com",
        "https://www.google.com",
        "https://www.paypal.com",
        "https://www.microsoft.com",
        "https://www.github.com",
        "https://www.wikipedia.org",
        "https://www.amazon.com",
        "https://facebook.com",
        "https://www.youtube.com",
        "http://192.168.1.1/admin/login.php",
        "http://secure-login-verify.example.com/account/update?id=123",
        "http://43.134.167.94/servicelogin?passive=1209600",
        "https://phishsense-frontend.vercel.app/",
    ]

    test_features = pd.DataFrame(
        [extract_features(url) for url in test_urls],
        columns=FEATURE_NAMES
    )
    test_scaled = scaler.transform(test_features)

    lr_probs = lr_model.predict_proba(test_scaled)
    xgb_probs = xgb_model.predict_proba(test_features)

    print(f"\n{'URL':<55} {'LR':>8} {'XGB':>8} {'LR Pred':>10} {'XGB Pred':>10}")
    print("-" * 95)

    for i, url in enumerate(test_urls):
        lr_phish = lr_probs[i][1]
        xgb_phish = xgb_probs[i][1]
        lr_label = "PHISH" if lr_phish > 0.5 else "LEGIT"
        xgb_label = "PHISH" if xgb_phish > 0.5 else "LEGIT"
        display_url = url[:53] + ".." if len(url) > 55 else url
        print(f"{display_url:<55} {lr_phish:>7.4f} {xgb_phish:>8.4f} {lr_label:>10} {xgb_label:>10}")

    # --- Feature importance ---
    print("\n=== XGBoost Feature Importance (Top 15) ===")
    importance = pd.Series(
        xgb_model.feature_importances_,
        index=FEATURE_NAMES
    ).sort_values(ascending=False)

    for feat, imp in importance.head(15).items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:<30} {imp:.4f} {bar}")

    # --- Save models ---
    output_dir = os.path.dirname(os.path.abspath(__file__))

    joblib.dump(lr_model, os.path.join(output_dir, "url_only_logistic_regression.pkl"))
    joblib.dump(xgb_model, os.path.join(output_dir, "url_only_xgboost.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "url_only_scaler.pkl"))
    joblib.dump(FEATURE_NAMES, os.path.join(output_dir, "url_only_feature_columns.pkl"))

    print(f"\n[+] Models saved to {output_dir}")
    print("  - url_only_logistic_regression.pkl")
    print("  - url_only_xgboost.pkl")
    print("  - url_only_scaler.pkl")
    print("  - url_only_feature_columns.pkl")
    print("\nDone! Restart the Flask app to use the new models.")


if __name__ == "__main__":
    main()
