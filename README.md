# 🛡️ ScamShield – Suspicious Message & Link Threat Detector
> **Two-Tier AI Threat Detection Engine for SMS, WhatsApp & Phishing Scams**
> *Built for rapid, production-quality cyber fraud defense.*

---

## 🚀 Key Highlights & Architecture

ScamShield implements a **Two-Tier Hybrid Threat Architecture**:

```
[ User Input (SMS / WhatsApp / Link) ]
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 1: Instant ML Risk Scan (Local Backend - <15ms)        │
│  • TF-IDF (1,2 n-grams) + Logistic Regression Pipeline     │
│  • Rule-Based Heuristic Threat Engine                      │
│    - URL Shortener Obfuscation (bit.ly, tinyurl, rb.gy)     │
│    - Insecure Protocol Flagging (HTTP vs HTTPS)             │
│    - Malicious / Phishing TLDs (.xyz, .top, .click, .tk)    │
│    - Psychological Urgency & Pressure Triggers             │
│    - Bank & Brand Impersonation Spoofing Checks             │
│    - Authentic OTP Disclaimers Safety Signals               │
│  • Output: Risk Score (0-100), Verdict, Red Flags, Telemetry│
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼ (User clicks "Run Deep Search")
┌─────────────────────────────────────────────────────────────┐
│  Tier 2: Grounded Live Truth Verification (Google Gemini)   │
│  • Google GenAI SDK: client.interactions.create             │
│  • Model: gemini-4-hydrogen                                │
│  • Live Google Search Grounding: Live Domain & FIR Check    │
│  • Forensic Breakdown: Psychological triggers & decoys     │
│  • Official Domain Comparison: Real bank/postal vs spoof   │
│  • Actionable Safety Protocol (Helpline 100, Chakshu)      │
│  • Clickable Grounded Web Citations Grid                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack
- **Backend:** Python 3.10+, FastAPI, Uvicorn, Scikit-Learn, Joblib, `google-genai` SDK, `python-dotenv`, Pydantic
- **Frontend:** Single-page `index.html` with Tailwind CSS (CDN), Marked.js, SVG Animated Risk Gauge, Glassmorphic Cyber UI
- **Storage:** Browser `localStorage` for persistent "Recent Scans History"

---

## 📁 Project Directory Structure
```
fraud detection/
├── backend/
│   ├── __init__.py
│   ├── train_model.py       # Curated 100-sample balanced dataset & TF-IDF training
│   ├── scam_model.joblib    # Serialized ML pipeline artifact (auto-generated)
│   └── main.py              # FastAPI server with /api/basic-scan & /api/deep-scan
├── frontend/
│   └── index.html           # Dark cyber glassmorphic UI + Marked.js + LocalStorage
├── .env.example             # Template for GEMINI_API_KEY
├── requirements.txt         # Project dependencies
└── README.md                # Documentation & Architecture Guide
```

---

## ⚡ Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for Deep Search)
Copy `.env.example` to `.env` and add your Gemini API Key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: You can also enter the Gemini API Key directly in the web UI via the "API Key" modal!)*

### 3. Train ML Model
```bash
python backend/train_model.py
```
*(The server also auto-trains the model on startup if `scam_model.joblib` is not present).*

### 4. Run Server
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 🧪 Demo Test Scenarios Included in UI

The app includes **4 One-Click Demo Buttons** to instantly test the system:

1. **🏦 Fake Bank KYC (`HIGH-RISK SCAM - 100/100`)**:
   > *"Dear SBI user, your YONO account has been suspended due to pending KYC. Update PAN immediately at http://sbi-kyc-verify.xyz/update or your account will be blocked within 24 hours."*
   - Flags: Insecure HTTP, `.xyz` phishing TLD, fake SBI domain, 24-hour urgency deadline, PAN demand.

2. **📦 Fake Courier Link (`HIGH-RISK SCAM - 82/100`)**:
   > *"India Post: Your package #IN89218 cannot be delivered due to incomplete street address. Please update your address within 12 hours: http://indiapost-parcel.top/update"*
   - Flags: Insecure HTTP, `.top` TLD, 12h deadline pressure, postal impersonation.

3. **🏛️ Fake Govt Scheme (`HIGH-RISK SCAM - 80/100`)**:
   > *"PM Mudra Loan Scheme 2026: Approved loan of ₹5,00,000 with 50% government subsidy and 0% interest. Register today: http://bit.ly/pm-mudra-yojana-apply"*
   - Flags: Obfuscated `bit.ly` shortener, loan subsidy lure, government impersonation.

4. **🔒 Genuine Bank OTP (`SAFE - 7/100`)**:
   > *"482910 is your secret OTP for HDFC Bank NetBanking transaction of INR 1,450.00 at Amazon India. Do NOT share this OTP with anyone, including bank staff."*
   - Recognized as legitimate OTP transaction with authentic safety disclaimers.

---

## 📡 API Endpoints

### 1. `POST /api/basic-scan`
**Request:**
```json
{
  "message": "Dear SBI user, your YONO account is suspended..."
}
```
**Response:**
```json
{
  "risk_score": 100,
  "verdict": "HIGH-RISK SCAM",
  "ml_probability": 0.7413,
  "urls_found": ["http://sbi-kyc-verify.xyz/update"],
  "red_flags": [
    "High urgency language detected",
    "Threat of account deactivation or block",
    "Unverified KYC / PAN / Identity update demand",
    "Artificial strict deadline / pressure tactic",
    "Potential spoofed bank domain detected: 'sbi-kyc-verify.xyz'",
    "Insecure HTTP link detected (unencrypted connection)",
    "Suspicious or abuse-prone TLD detected (.xyz, .top, .click, etc.)"
  ],
  "heuristics_summary": { ... }
}
```

### 2. `POST /api/deep-scan`
**Request:**
```json
{
  "message": "...",
  "basic_scan": { ... },
  "api_key": "optional_override_key"
}
```
**Response:**
```json
{
  "analysis": "### 1. Final Verdict & Verified Risk Score\n...",
  "citations": [
    { "title": "State Bank of India Official Portal", "url": "https://onlinesbi.sbi" }
  ],
  "model": "gemini-4-hydrogen"
}
```

---

## 🚨 Emergency Helplines Integrated
- **Police Emergency Helpline:** `100` (Toll-Free, 24/7 Police Emergency)
- **National Cybercrime Reporting Portal:** [cybercrime.gov.in](https://cybercrime.gov.in)
- **DoT Chakshu / Sanchar Saathi Portal:** [sancharsaathi.gov.in](https://sancharsaathi.gov.in)
