# NutriScan AI — Setup Guide

## Prerequisites
- Python 3.11+ installed
- Git installed
- Cursor (or VS Code) open on this project folder

---

## Step 1 — Get Your API Keys (do this first, takes ~5 minutes)

| Key | Where to get | Time |
|-----|-------------|------|
| **GROQ_API_KEY** | [console.groq.com](https://console.groq.com) → API Keys | 2 min |
| **USDA_API_KEY** | [api.nal.usda.gov](https://api.nal.usda.gov/api-key-signup) | 2 min (email approval) |

---

## Step 2 — Local Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv

# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env template and add your keys
cp .env.example .env
# Open .env and paste your API keys
```

---

## Step 3 — Add IFCT Indian Food Data

1. Go to: https://github.com/datameet/ifct2017
2. Download or clone the repo
3. Look for a JSON export (or convert the CSV using the script provided)
4. Place the file as: `data/ifct.json`

**If you skip this step:** The app still works — it falls back to USDA + Open Food Facts automatically. Indian food accuracy will be lower but functional.

---

## Step 4 — Run Tests First

```bash
python tests/test_scenarios.py
```

You should see mostly ✅ PASS. Fix any ❌ FAIL before proceeding.

Common failures:
- `GROQ_API_KEY is set` → Open .env and paste your key
- `Groq API connectivity` → Check your key is correct at console.groq.com
- `USDA lookup` → Your USDA key may not have activated yet (check email)

---

## Step 5 — Run the App Locally

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

**Enable Debug Mode** in the sidebar to see all API responses and internal state.

---

## Step 6 — GitHub Setup

```bash
# In your project folder:
git init
git add .
git commit -m "Initial scaffold — NutriScan AI"

# Create a new repo on github.com (empty, no README)
# Then:
git remote add origin https://github.com/YOUR_USERNAME/nutriscan-ai.git
git branch -M main
git push -u origin main
```

---

## Step 7 — Deploy to Streamlit Community Cloud (when ready to demo)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repo → `app.py` as main file
4. Click **Advanced settings** → Add Secrets:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   USDA_API_KEY = "your_usda_key"
   ```
5. Click **Deploy**
6. Share the public URL for your demo

> ⚠️ Never commit your `.env` file. It's in `.gitignore` already. Use Streamlit Secrets for deployment.

---

## Project Structure

```
nutriscan-ai/
├── app.py                    ← Main Streamlit app (start here)
├── requirements.txt          ← Python dependencies
├── .env.example              ← Copy to .env, add your keys
├── .gitignore
├── .streamlit/
│   └── config.toml           ← App theme and settings
├── utils/
│   ├── vision.py             ← Groq Llama 4 Scout: food detection from photos
│   ├── label_ocr.py          ← Groq Llama 4 Scout: nutrition label OCR
│   ├── portions.py           ← Converts "1 cup" → grams
│   ├── normalize.py          ← Maps "dal" → "lentils cooked" for DB lookup
│   ├── nutrition.py          ← IFCT / USDA / Open Food Facts lookup + aggregation
│   └── validate.py           ← 4-layer hallucination detection + disambiguation
├── data/
│   ├── README.md             ← Instructions for IFCT dataset
│   └── ifct.json             ← (download separately — see README.md)
└── tests/
    └── test_scenarios.py     ← Run this first to verify everything works
```

---

## Debug Tips

- **Enable Debug Mode** in the sidebar to see raw API responses
- All logs print to your terminal with timestamps — watch the terminal while testing
- If a food isn't recognized: check the normalize.py dict and add a mapping
- If nutrition data is wrong: check which `_source` is shown on the item card
- If Groq fails: check `USE_HF_FALLBACK=true` in .env to test the backup path

---

## Hackathon Demo Checklist

- [ ] All tests passing (`python tests/test_scenarios.py`)
- [ ] App runs locally without errors
- [ ] Meal photo flow tested with Indian thali photo
- [ ] Label OCR tested with a real packaged food label
- [ ] Disambiguation flow tested with a blurry/ambiguous photo
- [ ] Deployed to Streamlit Community Cloud
- [ ] Public URL working from a different device/network
- [ ] Debug Mode turned OFF before demo
- [ ] App opened 5 minutes before demo (warm-up)
- [ ] 3 demo scenarios prepared and rehearsed
