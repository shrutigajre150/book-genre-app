# BookLens — Deployment Guide
## How to get a public URL for your Streamlit app

---

## ✅ OPTION 1 — Streamlit Community Cloud (FREE, permanent public URL)
**Best for portfolios. Completely free.**

### Step 1 — Push to GitHub
```bash
# In your project folder:
git init
git add app.py requirements.txt
git commit -m "BookLens genre classifier"
git remote add origin https://github.com/YOUR_USERNAME/booklens.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **"Deploy"**

✅ You get a free public URL like: `https://your-username-booklens-app-abc123.streamlit.app`

---

## ✅ OPTION 2 — Run Locally + Share with ngrok (instant public URL)

### Step 1 — Install & run
```bash
pip install streamlit plotly pandas numpy scikit-learn scipy joblib
streamlit run app.py
# App runs on http://localhost:8501
```

### Step 2 — Make it public with ngrok
```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8501
# You get a public URL like: https://abc123.ngrok-free.app
```

---

## ✅ OPTION 3 — Google Colab (quick demo)
Run this in a Colab cell:
```python
!pip install streamlit pyngrok
!pip install plotly pandas scikit-learn scipy

# Save app.py (paste it in a cell first), then:
from pyngrok import ngrok
import subprocess, threading

def run():
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])

t = threading.Thread(target=run)
t.daemon = True
t.start()

import time; time.sleep(3)
url = ngrok.connect(8501)
print("🌐 Public URL:", url)
```

---

## 📁 Files in this folder

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app (all 5 tabs, EDA, ROC, Compare) |
| `requirements.txt` | Python dependencies |
| `models/` | Put your `.pkl` files here for live prediction |

## 🔑 To enable live prediction
Add these files to a `models/` folder next to `app.py`:
- `svm_model.pkl` — from `joblib.dump(best_model, "models/svm_model.pkl")`
- `lr_model.pkl`  — from `joblib.dump(lr_pipeline, "models/lr_model.pkl")`
- `label_encoder.pkl` — from `joblib.dump(le, "models/label_encoder.pkl")`

In your Colab notebook, add this after training:
```python
import joblib, os
os.makedirs("models", exist_ok=True)
joblib.dump(best_model,   "models/svm_model.pkl")
joblib.dump(lr_pipeline,  "models/lr_model.pkl")
joblib.dump(le,           "models/label_encoder.pkl")
```
Then download these 3 files from Colab and place them in your `models/` folder.

## ✏️ Personalise the app
In `app.py`, find these lines and update them:
- Line ~190: `Your Name` → your actual name  
- Line ~193: `https://github.com` → your GitHub URL
- Line ~194: `https://linkedin.com` → your LinkedIn URL
- Line ~195: `your@email.com` → your email
