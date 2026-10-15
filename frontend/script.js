const urlInput = document.getElementById('urlInput');
const scanBtn = document.getElementById('scanBtn');
const resultsSection = document.getElementById('results');
const resultBody = document.getElementById('resultBody');
const resultDetails = document.getElementById('resultDetails');
const closeResultBtn = document.getElementById('closeResult');
const scanAgainBtn = document.getElementById('scanAgainBtn');

const navLinks = document.querySelectorAll('.nav-link');
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

// Production backend on Render
const API_URL = 'https://phishsense-api.onrender.com/predict-url';

function normalizeUrl(url) {
    url = url.trim();
    if (!/^https?:\/\//i.test(url)) {
        url = 'https://' + url;
    }
    return url;
}

function getDomainInfo(url) {
    const parsed = new URL(url);
    return {
        domain: parsed.hostname,
        protocol: parsed.protocol,
        path: parsed.pathname
    };
}

async function scanUrl() {
    const url = urlInput.value.trim();

    if (!url) {
        showError('Please enter a URL.');
        return;
    }

    const normalizedUrl = normalizeUrl(url);

    try {
        new URL(normalizedUrl);
    } catch {
        showError('Please enter a valid URL.');
        return;
    }

    const btnText = scanBtn.querySelector('.btn-text');
    const btnLoader = scanBtn.querySelector('.btn-loader');

    scanBtn.disabled = true;
    if (btnText) btnText.style.display = 'none';
    if (btnLoader) btnLoader.style.display = 'inline-block';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: normalizedUrl })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'URL scan failed.');
        }

        displayResults(normalizedUrl, data, getDomainInfo(normalizedUrl));

    } catch (error) {
        console.error('Scan error:', error);
        showError(
            error.message === 'Failed to fetch'
                ? 'Cannot connect to the backend. Make sure Flask is running.'
                : error.message
        );
    } finally {
        scanBtn.disabled = false;
        if (btnText) btnText.style.display = 'inline';
        if (btnLoader) btnLoader.style.display = 'none';
    }
}

function displayResults(url, data, domainInfo) {
    const lr = data.logistic_regression;
    const xgb = data.xgboost;
    const isTrusted = data.trusted_domain || false;

    const lrPhishing = lr.prediction === 'Phishing';
    const xgbPhishing = xgb.prediction === 'Phishing';

    let level;
    let color;
    let icon;
    let message;

    if (lrPhishing === xgbPhishing) {
        if (lrPhishing) {
            level = 'POTENTIALLY DANGEROUS';
            color = '#ef4444';
            icon = 'fa-times-circle';
            message = 'Both models flagged this URL as potentially suspicious.';
        } else {
            level = 'LOWER RISK';
            color = '#10b981';
            icon = 'fa-check-circle';
            message = isTrusted
                ? 'This is a verified trusted domain. Both models confirm it as legitimate.'
                : 'Both models classified this URL as legitimate. This does not guarantee safety.';
        }
    } else {
        level = 'UNCERTAIN';
        color = '#f97316';
        icon = 'fa-exclamation-triangle';
        message = 'The models disagree. Treat this URL cautiously.';
    }

    resultsSection.style.display = 'flex';

    const trustedBadge = isTrusted
        ? `<div style="margin-top:0.5rem;">
               <span style="display:inline-flex;align-items:center;gap:0.35rem;background:rgba(16,185,129,0.15);color:#10b981;padding:0.3rem 0.8rem;border-radius:999px;font-size:0.85rem;font-weight:600;">
                   <i class="fas fa-shield-alt"></i> Trusted Domain
               </span>
           </div>`
        : '';

    resultBody.innerHTML = `
        <div style="text-align:center;">
            <i class="fas ${icon}" style="font-size:3rem;color:${color};margin-bottom:1rem;"></i>
            <span class="status-badge ${lrPhishing === xgbPhishing ? (lrPhishing ? 'status-danger' : 'status-safe') : 'status-suspicious'}">${level}</span>
            <p style="font-size:1.1rem;margin:0.5rem 0;color:var(--text-secondary);">${message}</p>
            ${trustedBadge}
            <div class="result-url"></div>
        </div>
    `;

    resultBody.querySelector('.result-url').textContent = url;

    const lrPercent = (lr.phishing_probability * 100).toFixed(2);
    const xgbPercent = (xgb.phishing_probability * 100).toFixed(2);

    resultDetails.innerHTML = `
        <div style="margin-top:1.5rem;">
            <h3 style="margin-bottom:1rem;color:var(--text-primary);">Model Analysis</h3>
            <div class="detail-item">
                <span class="detail-label">Logistic Regression</span>
                <span class="detail-value">${lr.prediction} (${lrPercent}% phishing score)</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">XGBoost</span>
                <span class="detail-value">${xgb.prediction} (${xgbPercent}% phishing score)</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Domain</span>
                <span class="detail-value" style="font-family:monospace;"></span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Protocol</span>
                <span class="detail-value">${domainInfo.protocol}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Scan Date</span>
                <span class="detail-value">${new Date().toLocaleString()}</span>
            </div>
            <div class="detail-item" style="border-bottom:none;">
                <span class="detail-label">Method</span>
                <span class="detail-value">URL text only</span>
            </div>
            <p style="font-size:0.85rem;color:var(--text-secondary);margin-top:1rem;">
                This is a machine-learning prediction, not a guarantee that a website is safe or malicious.
            </p>
        </div>
    `;

    resultDetails.querySelector('.detail-item:nth-of-type(3) .detail-value').textContent = domainInfo.domain;

    setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth' }), 100);
}

function showError(message) {
    resultsSection.style.display = 'flex';
    resultBody.innerHTML = '';

    const errorText = document.createElement('p');
    errorText.style.cssText = 'text-align:center;color:#ef4444;font-size:1.1rem;';
    errorText.textContent = message;
    resultBody.appendChild(errorText);

    resultDetails.innerHTML = '';
    setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth' }), 100);
}

function closeResults() {
    resultsSection.style.display = 'none';
    urlInput.focus();
}

function scanAgain() {
    urlInput.value = '';
    resultsSection.style.display = 'none';
    urlInput.focus();
}

// Enable scan interactions
scanBtn.addEventListener('click', scanUrl);

urlInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') scanUrl();
});

closeResultBtn?.addEventListener('click', closeResults);
scanAgainBtn?.addEventListener('click', scanAgain);

// Navigation
navLinks.forEach(link => {
    link.addEventListener('click', () => {
        navLinks.forEach(item => item.classList.remove('active'));
        link.classList.add('active');
        navMenu?.classList.remove('active');
    });
});

hamburger?.addEventListener('click', () => {
    navMenu?.classList.toggle('active');
});

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (event) {
        const href = this.getAttribute('href');
        if (!href || href === '#') return;

        const target = document.querySelector(href);
        if (target) {
            event.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// Prevent form reload
document.querySelector('form')?.addEventListener('submit', event => {
    event.preventDefault();
    scanUrl();
});

document.addEventListener('DOMContentLoaded', () => {
    console.log('PhishSense initialized');
    urlInput?.focus();
});

// Dark mode preference
if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
    document.body.classList.add('dark-mode');
}

window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener('change', event => {
    document.body.classList.toggle('dark-mode', event.matches);
});
