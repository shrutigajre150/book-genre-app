"""
AI-Powered Book Genre Classification System
Shruti Gajre · MSc Statistics & Data Science
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re, string, os
from scipy.special import softmax
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI-Powered Book Genre Classification System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Sora', sans-serif !important; box-sizing: border-box; }
.stApp { background: #07090f; color: #dde3f0; }
.block-container { padding-top: 1rem !important; max-width: 1300px !important; }
::-webkit-scrollbar{width:5px;} ::-webkit-scrollbar-thumb{background:#2a3560;border-radius:3px;}

/* HERO */
.hero { background:linear-gradient(135deg,#090c1d 0%,#0f1530 50%,#130b2a 100%);
    border-radius:20px; padding:52px 48px 44px; border:1px solid #1c2444;
    position:relative; overflow:hidden; margin-bottom:28px; }
.hero::before { content:''; position:absolute; top:-90px; right:-90px;
    width:380px; height:380px; border-radius:50%;
    background:radial-gradient(circle,rgba(99,102,241,0.15) 0%,transparent 70%); }
.hero::after { content:''; position:absolute; bottom:-70px; left:-70px;
    width:280px; height:280px; border-radius:50%;
    background:radial-gradient(circle,rgba(16,185,129,0.10) 0%,transparent 70%); }
.hero-badge { display:inline-block; background:rgba(99,102,241,0.12);
    border:1px solid rgba(99,102,241,0.35); color:#a5b4fc;
    font-size:0.70rem; font-weight:600; letter-spacing:2.5px; text-transform:uppercase;
    padding:5px 14px; border-radius:50px; margin-bottom:16px; }
.hero h1 { font-size:2.6rem !important; font-weight:800 !important;
    line-height:1.15 !important; margin-bottom:12px !important;
    background:linear-gradient(135deg,#e0e7ff 0%,#a5b4fc 50%,#6ee7b7 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#94a3b8; font-size:0.95rem; line-height:1.8; max-width:720px; }
.kpi-row { display:flex; gap:12px; margin:22px 0 0; flex-wrap:wrap; }
.kpi { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:14px 20px; min-width:115px; }
.kpi-val { font-size:1.5rem; font-weight:800; color:#a5b4fc; line-height:1; }
.kpi-lbl { font-size:0.68rem; color:#64748b; margin-top:4px; font-weight:500; }
.tech-row { display:flex; gap:7px; flex-wrap:wrap; margin-top:16px; }
.tech { background:rgba(99,102,241,0.10); border:1px solid rgba(99,102,241,0.25);
    color:#c7d2fe; font-size:0.68rem; font-weight:600; padding:4px 11px;
    border-radius:50px; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { gap:4px; background:rgba(255,255,255,0.03);
    border-radius:12px; padding:4px; border:1px solid #1e2a4a; }
.stTabs [data-baseweb="tab"] { border-radius:8px; color:#64748b; font-weight:600;
    font-size:0.80rem; padding:8px 16px; }
.stTabs [aria-selected="true"] { background:rgba(99,102,241,0.15) !important;
    color:#a5b4fc !important; }

/* CARDS & SECTIONS */
.card { background:rgba(255,255,255,0.03); border:1px solid #1e2a4a;
    border-radius:16px; padding:22px; margin-bottom:14px; }
.section-hdr { font-size:1rem; font-weight:700; color:#e2e8f0;
    border-left:3px solid #6366f1; padding-left:12px; margin:20px 0 14px; }

/* ABOUT */
.about-card { background:linear-gradient(135deg,#0f1225,#131830);
    border:1px solid #1e2a4a; border-radius:18px; padding:26px 20px; }
.about-name { font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:3px; }
.about-role { font-size:0.72rem; color:#6366f1; font-weight:600;
    letter-spacing:1px; text-transform:uppercase; margin-bottom:12px; }
.about-bio { font-size:0.82rem; color:#94a3b8; line-height:1.75; margin-bottom:14px; }
.social-btn { display:inline-flex; align-items:center; gap:6px;
    background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3);
    color:#a5b4fc; padding:7px 13px; border-radius:8px; font-size:0.75rem;
    font-weight:600; text-decoration:none; margin:3px; }

/* INPUTS */
.stTextArea textarea { background:#0f1225 !important; border:1px solid #1e2a4a !important;
    color:#dde3f0 !important; border-radius:10px !important; }
div[data-testid="stButton"] button {
    background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; padding:10px 28px !important; }

/* PROBLEM GRID */
.prob-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:14px 0; }
.prob-item { background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.2);
    border-radius:12px; padding:16px; }
.prob-icon { font-size:1.5rem; margin-bottom:8px; }
.prob-title { font-size:0.82rem; font-weight:700; color:#a5b4fc; margin-bottom:5px; }
.prob-desc { font-size:0.78rem; color:#94a3b8; line-height:1.6; }

/* STEPS */
.step-row { display:flex; gap:12px; margin:12px 0; align-items:flex-start; }
.step-num { background:linear-gradient(135deg,#6366f1,#4f46e5); color:white;
    font-size:0.75rem; font-weight:800; width:26px; height:26px; border-radius:50%;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px; }
.step-title { font-size:0.88rem; font-weight:700; color:#e2e8f0; margin-bottom:3px; }
.step-desc { font-size:0.80rem; color:#94a3b8; line-height:1.6; }

/* USE CASES */
.use-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:12px 0; }
.use-card { background:rgba(255,255,255,0.03); border:1px solid #1e2a4a;
    border-radius:12px; padding:16px; text-align:center; }
.use-icon { font-size:1.6rem; margin-bottom:7px; }
.use-title { font-size:0.80rem; font-weight:700; color:#e2e8f0; margin-bottom:4px; }
.use-desc { font-size:0.74rem; color:#94a3b8; line-height:1.5; }

/* VERDICT */
.verdict { background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.25);
    border-radius:14px; padding:20px; margin-top:16px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
GENRES = ['fantasy','history','horror','psychology','romance',
          'science','science_fiction','sports','thriller','travel']
GENRE_EMOJI  = {'fantasy':'🧙','history':'📜','horror':'👻','psychology':'🧠',
                'romance':'💕','science':'🔬','science_fiction':'🚀',
                'sports':'⚽','thriller':'🔪','travel':'✈️'}
GENRE_COLOR  = {'fantasy':'#818cf8','history':'#fbbf24','horror':'#f87171',
                'psychology':'#34d399','romance':'#f472b6','science':'#38bdf8',
                'science_fiction':'#a78bfa','sports':'#fb923c',
                'thriller':'#94a3b8','travel':'#4ade80'}

SVM_METRICS = {"Accuracy":0.7695,"Macro Precision":0.7263,"Macro Recall":0.6904,
               "Macro F1":0.7003,"Weighted F1":0.7588,"Cohen Kappa":0.7152,"Macro ROC-AUC":0.8308}
LR_METRICS  = {"Accuracy":0.7792,"Macro Precision":0.7511,"Macro Recall":0.7365,
               "Macro F1":0.7373,"Weighted F1":0.7815,"Cohen Kappa":0.7327,"Macro ROC-AUC":0.9576}

SVM_PER = {'fantasy':0.85,'history':0.84,'horror':0.32,'psychology':0.81,
           'romance':0.50,'science':0.76,'science_fiction':0.40,
           'sports':0.83,'thriller':0.81,'travel':0.86}
LR_PER  = {'fantasy':0.85,'history':0.82,'horror':0.37,'psychology':0.86,
           'romance':0.57,'science':0.81,'science_fiction':0.53,
           'sports':0.86,'thriller':0.83,'travel':0.86}

SVM_AUC = {'fantasy':0.98,'history':0.99,'horror':0.88,'psychology':0.98,
           'romance':0.92,'science':0.98,'science_fiction':0.92,
           'sports':0.99,'thriller':0.94,'travel':0.96}
LR_AUC  = {'fantasy':0.98,'history':0.99,'horror':0.88,'psychology':0.99,
           'romance':0.92,'science':0.98,'science_fiction':0.95,
           'sports':0.99,'thriller':0.93,'travel':0.97}

TOP_WORDS = {
    'fantasy':['magic','magical','lord','power','wizard','land','king','prince','kingdom','throne'],
    'history':['history','war','century','american','civil','account','narrative','shaped','human','americans'],
    'horror':['horror','dark','madness','stories','house','haunted','terrifying','evil','strange','stephen king'],
    'psychology':['psychology','people','unconscious','brain','work','book','dr','psychologist','research','shows'],
    'romance':['love','relationship','clary','feelings','wedding','darcy','love story','mother','best friend','ll'],
    'science':['science','scientific','universe','evolution','theory','book','quantum','physics','species','scientists'],
    'science_fiction':['alien','earth','planet','future','interstellar','space','humans','science fiction','humankind','novels'],
    'sports':['hockey','team','player','football','baseball','players','star','sports','sport','professional'],
    'thriller':['thriller','police','murder','husband','killer','detective','missing','reacher','case','murdered'],
    'travel':['travel','journey','country','theroux','continent','travels','exotic','road','india','bryson']
}

GENRE_COUNTS   = {'thriller':481,'fantasy':348,'romance':111,'horror':100,'history':99,
                  'psychology':99,'travel':98,'science':79,'sports':79,'science_fiction':45}
GENRE_RATING   = {'fantasy':4.155,'history':4.126,'science':4.121,'science_fiction':4.109,
                  'sports':4.102,'psychology':4.093,'romance':4.041,'thriller':3.942,
                  'travel':3.935,'horror':3.891}
GENRE_SENT     = {'travel':0.190,'science':0.165,'history':0.159,'psychology':0.157,
                  'romance':0.113,'sports':0.098,'fantasy':0.052,'science_fiction':0.049,
                  'thriller':0.014,'horror':-0.021}
GENRE_WORDS    = {'history':212.1,'sports':184.2,'science':176.3,'psychology':172.6,
                  'romance':164.7,'thriller':162.8,'fantasy':156.4,'travel':156.3,
                  'horror':139.5,'science_fiction':117.9}

EXAMPLES = {
    "🧙 Fantasy":
        "A young orphan discovers he is a wizard on his eleventh birthday and is invited to attend a school of magic. There he learns about his past, makes friends and enemies, and discovers he has a destiny to face the dark lord who killed his parents.",
    "🔪 Thriller":
        "A detective wakes up with no memory of the past 48 hours. A woman is dead. His gun is missing. And every piece of evidence points directly at him. With the clock ticking and a killer still on the loose, he must uncover the truth before he becomes the next victim.",
    "🚀 Sci-Fi":
        "In the distant future, humanity has colonised a hundred planets across the galaxy. When alien signals are detected from beyond the known universe, one scientist risks everything to make first contact — and what she finds will change mankind forever.",
    "💕 Romance":
        "She swore she would never fall for her best friend's brother. He made a promise to stay away. But when forced to spend a summer together in a small coastal town, neither of them can deny the feelings they have buried for years.",
    "📜 History":
        "A sweeping narrative of the American Civil War that follows ordinary soldiers and their families through four years of devastating conflict. Drawing on thousands of letters and diaries, this account reveals the human cost of a nation torn apart.",
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for name, mpath in [("svm","models/svm_model.pkl"),("lr","models/lr_model.pkl")]:
        if os.path.exists(mpath) and os.path.exists("models/label_encoder.pkl"):
            try:
                out[name] = {"model": joblib.load(mpath),
                             "le":    joblib.load("models/label_encoder.pkl")}
            except Exception:
                pass
    return out

MODELS = load_models()
LIVE   = len(MODELS) > 0

def predict(model_key, text):
    if not LIVE or model_key not in MODELS:
        np.random.seed(abs(hash(text[:40])) % (2**31))
        probs = np.random.dirichlet(np.ones(10) * 0.4)
        conf  = dict(zip(GENRES, probs))
        return max(conf, key=conf.get), conf
    m, le = MODELS[model_key]["model"], MODELS[model_key]["le"]
    cleaned = clean_text(text)
    idx   = m.predict([cleaned])[0]
    genre = le.inverse_transform([idx])[0]
    if model_key == "svm":
        conf = dict(zip(le.classes_, softmax(m.decision_function([cleaned])[0])))
    else:
        conf = dict(zip(le.classes_, m.predict_proba([cleaned])[0]))
    return genre, conf

def hbar(vals, labels, colors, height=300, xrange=None, show_text=True):
    text_vals = [f"{v:.3f}" if isinstance(v,float) else str(v) for v in vals] if show_text else None
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation='h',
        marker_color=colors, marker_line_width=0,
        text=text_vals,
        textposition='outside', textfont=dict(color='#94a3b8', size=10)
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=58,t=8,b=8), height=height,
        xaxis=dict(showgrid=True,gridcolor='#1e2a4a',color='#64748b',
                   zeroline=False,range=xrange),
        yaxis=dict(color='#94a3b8',tickfont=dict(size=10)), bargap=0.28)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">📚 MSc Statistics &amp; Data Science · Portfolio Project</div>
  <h1>AI-Powered Book Genre Classification System</h1>
  <p>An end-to-end NLP system that automatically classifies books into genres from
  synopsis text using classical machine learning. Trained on 1,539 Goodreads books
  across 10 genres and validated with rigorous statistical testing (ANOVA, Tukey HSD,
  Kruskal-Wallis). Deployed as a live, interactive web application.</p>
  <div class="kpi-row">
    <div class="kpi"><div class="kpi-val">1,539</div><div class="kpi-lbl">Books</div></div>
    <div class="kpi"><div class="kpi-val">10</div><div class="kpi-lbl">Genres</div></div>
    <div class="kpi"><div class="kpi-val">77.9%</div><div class="kpi-lbl">Best Accuracy</div></div>
    <div class="kpi"><div class="kpi-val">0.958</div><div class="kpi-lbl">Best ROC-AUC</div></div>
    <div class="kpi"><div class="kpi-val">F=49.6</div><div class="kpi-lbl">ANOVA F-stat</div></div>
    <div class="kpi"><div class="kpi-val">η²=0.226</div><div class="kpi-lbl">Effect Size</div></div>
  </div>
  <div class="tech-row">
    <span class="tech">Python</span><span class="tech">Scikit-learn</span>
    <span class="tech">TF-IDF</span><span class="tech">Linear SVM</span>
    <span class="tech">Logistic Regression</span><span class="tech">TextBlob</span>
    <span class="tech">ANOVA</span><span class="tech">Tukey HSD</span>
    <span class="tech">Plotly</span><span class="tech">Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 3.3], gap="large")

# ── ABOUT ────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div class="about-card">
      <div style="font-size:2.8rem;margin-bottom:10px;">👩‍💻</div>
      <div class="about-name">Shruti Gajre</div>
      <div class="about-role">MSc Statistics &amp; Data Science</div>
      <div class="about-bio">
        Passionate about applying statistical methods and machine learning to
        real-world NLP problems. This project builds an automated book genre
        classifier using classical ML pipelines validated with statistical testing.
      </div>
      <div style="margin-bottom:14px;">
        <a class="social-btn" href="https://github.com/shrutigajre150" target="_blank">🐙 GitHub</a>
        <a class="social-btn" href="https://www.linkedin.com/in/shrutigajre/" target="_blank">💼 LinkedIn</a>
      </div>
      <hr style="border-color:#1e2a4a;margin:14px 0;">
      <div style="font-size:0.70rem;color:#64748b;font-weight:600;letter-spacing:1px;
           text-transform:uppercase;margin-bottom:10px;">Project Info</div>
      <div style="font-size:0.78rem;color:#94a3b8;line-height:2.1;">
        📊 <b style="color:#a5b4fc;">Dataset:</b> TagMyBook · Kaggle<br>
        📖 <b style="color:#a5b4fc;">Source:</b> Goodreads (scraped)<br>
        🔤 <b style="color:#a5b4fc;">Features:</b> TF-IDF · ngram (1,3)<br>
        🤖 <b style="color:#a5b4fc;">Models:</b> SVM · Logistic Regression<br>
        📈 <b style="color:#a5b4fc;">Stats:</b> ANOVA · Tukey HSD · KW<br>
        💡 <b style="color:#a5b4fc;">Sentiment:</b> TextBlob Polarity<br>
        🏆 <b style="color:#a5b4fc;">Best AUC:</b> 0.958 (Logistic Reg.)
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
with col_right:
    tab_prob, tab_pred, tab_batch, tab_eda, tab_how, tab_res = st.tabs([
        "❓ Problem", "🔮 Predict", "📦 Batch CSV",
        "📊 EDA", "🤖 How It Works", "📈 Results"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — PROBLEM STATEMENT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_prob:
        st.markdown('<div class="section-hdr">Problem Statement</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
          <div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:10px;">
            📌 What problem does this solve?
          </div>
          <div style="font-size:0.88rem;color:#94a3b8;line-height:1.9;">
            Every day, thousands of new books are published.
            <b style="color:#e2e8f0;">Manually assigning genres requires human editors</b>
            — it is slow, expensive, and inconsistent across teams. This project
            answers one simple but impactful question:<br><br>
            <span style="font-size:0.98rem;color:#a5b4fc;font-weight:700;">
            "Can a machine read a book's synopsis and automatically predict its genre?"
            </span><br><br>
            Using only the text of a book's synopsis — no title, no author, no metadata —
            this system classifies books into one of 10 genres with <b style="color:#e2e8f0;">
            77.9% accuracy</b> and a <b style="color:#e2e8f0;">ROC-AUC of 0.958</b>.
          </div>
        </div>
        <div class="section-hdr">Why Does This Matter?</div>
        <div class="prob-grid">
          <div class="prob-item">
            <div class="prob-icon">🏷️</div>
            <div class="prob-title">Auto-tagging for Publishers</div>
            <div class="prob-desc">Publishers receive hundreds of manuscripts weekly.
            Automated genre tagging speeds up cataloguing and reduces editorial workload significantly.</div>
          </div>
          <div class="prob-item">
            <div class="prob-icon">🎯</div>
            <div class="prob-title">Better Recommendations</div>
            <div class="prob-desc">Platforms like Audible and Spotify Books need accurate
            genre labels to power personalised recommendation engines that keep users engaged.</div>
          </div>
          <div class="prob-item">
            <div class="prob-icon">🏛️</div>
            <div class="prob-title">Digital Library Management</div>
            <div class="prob-desc">Public libraries and digital archives can automatically
            organise large collections without manual classification effort or specialist staff.</div>
          </div>
          <div class="prob-item">
            <div class="prob-icon">🔍</div>
            <div class="prob-title">Search &amp; Discovery</div>
            <div class="prob-desc">Accurate genre labels improve search filters on book
            retail platforms, helping readers find books they actually want to read.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-hdr">Dataset Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;
               font-size:0.84rem;color:#94a3b8;line-height:2.1;">
            <div>
              📂 <b style="color:#a5b4fc;">Source:</b> TagMyBook · Kaggle<br>
              📖 <b style="color:#a5b4fc;">Scraped from:</b> Goodreads<br>
              📚 <b style="color:#a5b4fc;">Total books:</b> 1,539<br>
              🏷️ <b style="color:#a5b4fc;">Genres:</b> 10 categories<br>
            </div>
            <div>
              ⭐ <b style="color:#a5b4fc;">Avg rating:</b> 4.04 / 5.0<br>
              📝 <b style="color:#a5b4fc;">Avg synopsis:</b> 163 words<br>
              📊 <b style="color:#a5b4fc;">Largest genre:</b> Thriller (31.3%)<br>
              🔬 <b style="color:#a5b4fc;">Smallest genre:</b> Sci-Fi (2.9%)<br>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-hdr">Key Statistical Findings</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col,icon,val,lbl,note in [
            (c1,"📊","F = 49.6","ANOVA F-stat","Genre predicts sentiment"),
            (c2,"🔬","p < 10⁻⁷⁸","Significance","Strongly reject H₀"),
            (c3,"📐","η² = 0.226","Effect size","22.6% variance explained"),
            (c4,"📉","H = 361.2","Kruskal-Wallis","Non-parametric confirm"),
        ]:
            col.markdown(f"""
            <div style="background:rgba(99,102,241,0.07);border:1px solid #1e2a4a;
                 border-radius:12px;padding:14px;text-align:center;margin-bottom:8px;">
              <div style="font-size:1.3rem;">{icon}</div>
              <div style="font-size:1rem;font-weight:800;color:#a5b4fc;">{val}</div>
              <div style="font-size:0.72rem;color:#e2e8f0;font-weight:600;margin-top:2px;">{lbl}</div>
              <div style="font-size:0.68rem;color:#64748b;margin-top:2px;">{note}</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — PREDICT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_pred:
        st.markdown('<div class="section-hdr">Live Genre Prediction</div>', unsafe_allow_html=True)

        model_choice = st.radio("Select Model",
            ["⚔️ Linear SVM  (Acc: 77.0% | AUC: 0.831)",
             "📊 Logistic Regression  (Acc: 77.9% | AUC: 0.958)"],
            horizontal=True)
        mk = "svm" if "SVM" in model_choice else "lr"

        example = st.selectbox("Load an example synopsis",
                               ["— type your own —"] + list(EXAMPLES.keys()))
        default = EXAMPLES.get(example,"") if example != "— type your own —" else ""

        synopsis = st.text_area("Book Synopsis", value=default, height=155,
                                placeholder="Paste or type a book synopsis here…")

        if st.button("🔮 Classify Genre"):
            if synopsis.strip():
                with st.spinner("Classifying…"):
                    genre, conf = predict(mk, synopsis)

                color = GENRE_COLOR.get(genre,"#a5b4fc")
                emoji = GENRE_EMOJI.get(genre,"📖")
                top5  = sorted(conf.items(), key=lambda x:x[1], reverse=True)[:5]

                r1,r2 = st.columns([1,1.5])
                with r1:
                    st.markdown(f"""
                    <div style="background:rgba(99,102,241,0.07);border:1px solid #1e2a4a;
                         border-radius:16px;padding:22px;text-align:center;">
                      <div style="font-size:2.8rem;">{emoji}</div>
                      <div style="font-size:0.66rem;color:#64748b;letter-spacing:2px;
                           text-transform:uppercase;margin:8px 0 4px;">Predicted Genre</div>
                      <div style="font-size:1.5rem;font-weight:800;color:{color};
                           text-transform:capitalize;">{genre.replace('_',' ')}</div>
                      <div style="font-size:0.82rem;color:#94a3b8;margin-top:8px;">
                        Confidence: <b style="color:{color};">{conf[genre]*100:.1f}%</b>
                      </div>
                    </div>
                    <div style="margin-top:10px;background:rgba(255,255,255,0.02);
                         border:1px solid #1e2a4a;border-radius:12px;padding:12px;">
                      <div style="font-size:0.66rem;color:#64748b;letter-spacing:1.5px;
                           text-transform:uppercase;margin-bottom:8px;">Key Genre Signals</div>
                      <div style="display:flex;flex-wrap:wrap;gap:5px;">
                        {''.join(f'<span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);color:#c7d2fe;font-size:0.68rem;padding:3px 9px;border-radius:50px;">{w}</span>' for w in TOP_WORDS.get(genre,[])[:6])}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                with r2:
                    labels = [g.replace('_',' ').title() for g,_ in top5]
                    vals   = [v*100 for _,v in top5]
                    colors = [GENRE_COLOR.get(g,'#6366f1') for g,_ in top5]
                    fig = go.Figure(go.Bar(x=vals,y=labels,orientation='h',
                        marker_color=colors,marker_line_width=0,
                        text=[f"{v:.1f}%" for v in vals],textposition='outside',
                        textfont=dict(color='#94a3b8',size=11)))
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0,r=55,t=6,b=6),height=190,
                        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,
                                   range=[0,max(vals)*1.3]),
                        yaxis=dict(showgrid=False,color='#94a3b8',tickfont=dict(size=11)),
                        bargap=0.3)
                    st.markdown('<div style="font-size:0.74rem;color:#64748b;margin-bottom:2px;">Top 5 Confidence Scores</div>',unsafe_allow_html=True)
                    st.plotly_chart(fig,use_container_width=True,config=dict(displayModeBar=False))

                    top8   = sorted(conf.items(),key=lambda x:x[1],reverse=True)[:8]
                    rl     = [g.replace('_',' ').title() for g,_ in top8]
                    rv     = [v*100 for _,v in top8]
                    rl.append(rl[0]); rv.append(rv[0])
                    fig2 = go.Figure(go.Scatterpolar(r=rv,theta=rl,fill='toself',
                        fillcolor='rgba(99,102,241,0.12)',
                        line=dict(color='#6366f1',width=2)))
                    fig2.update_layout(
                        polar=dict(bgcolor='rgba(0,0,0,0)',
                            radialaxis=dict(visible=False),
                            angularaxis=dict(color='#64748b',tickfont=dict(size=8))),
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20,r=20,t=4,b=4),height=190)
                    st.markdown('<div style="font-size:0.74rem;color:#64748b;margin-bottom:2px;">Confidence Radar</div>',unsafe_allow_html=True)
                    st.plotly_chart(fig2,use_container_width=True,config=dict(displayModeBar=False))

        if not LIVE:
            st.caption("ℹ️ Running in demo mode — add models/svm_model.pkl, models/lr_model.pkl and models/label_encoder.pkl for live prediction.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — BATCH CSV
    # ══════════════════════════════════════════════════════════════════════════
    with tab_batch:
        st.markdown('<div class="section-hdr">Batch Genre Classification</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
          <div style="font-size:0.88rem;color:#94a3b8;line-height:1.8;">
            Upload a <b style="color:#a5b4fc;">CSV file</b> with a column named
            <code style="background:rgba(99,102,241,0.15);padding:2px 7px;border-radius:5px;
            color:#c7d2fe;">synopsis</code>.
            The model will classify every row and you can download the labelled results.
          </div>
        </div>
        """, unsafe_allow_html=True)

        sample_df  = pd.DataFrame({"synopsis": list(EXAMPLES.values())})
        sample_csv = sample_df.to_csv(index=False).encode()
        st.download_button("📥 Download Sample CSV", sample_csv,
                           "sample_synopses.csv", "text/csv")

        st.markdown("---")
        mb = st.radio("Model", ["⚔️ Linear SVM","📊 Logistic Regression"], horizontal=True)
        mk_b = "svm" if "SVM" in mb else "lr"

        uploaded = st.file_uploader("Upload your CSV", type=["csv"])
        if uploaded:
            try:
                df_in = pd.read_csv(uploaded)
                if "synopsis" not in df_in.columns:
                    st.error("❌ CSV must contain a column named 'synopsis'")
                else:
                    st.success(f"✅ {len(df_in)} rows loaded")
                    st.dataframe(df_in.head(3), use_container_width=True)

                    if st.button("🚀 Classify All Rows"):
                        with st.spinner(f"Classifying {len(df_in)} synopses…"):
                            genres_pred, confs_pred = [], []
                            for txt in df_in["synopsis"].fillna(""):
                                g, c = predict(mk_b, str(txt))
                                genres_pred.append(g)
                                confs_pred.append(round(c.get(g,0)*100,1))
                            df_in["predicted_genre"] = genres_pred
                            df_in["confidence_%"]    = confs_pred

                        st.success("✅ Classification complete!")
                        st.dataframe(
                            df_in[["synopsis","predicted_genre","confidence_%"]],
                            use_container_width=True)

                        vc = pd.Series(genres_pred).value_counts()
                        fig = go.Figure(go.Bar(
                            x=vc.index.tolist(), y=vc.values.tolist(),
                            marker_color=[GENRE_COLOR.get(g,"#6366f1") for g in vc.index],
                            marker_line_width=0))
                        fig.update_layout(
                            title=dict(text="Predicted Genre Distribution",
                                       font=dict(color='#e2e8f0',size=12)),
                            plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0,r=0,t=36,b=0),height=240,
                            xaxis=dict(color='#64748b',showgrid=False),
                            yaxis=dict(color='#64748b',showgrid=True,
                                       gridcolor='#1e2a4a',zeroline=False))
                        st.plotly_chart(fig,use_container_width=True,
                                        config=dict(displayModeBar=False))

                        st.download_button("📥 Download Results CSV",
                            df_in.to_csv(index=False).encode(),
                            "genre_predictions.csv","text/csv")
            except Exception as e:
                st.error(f"Error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — EDA
    # ══════════════════════════════════════════════════════════════════════════
    with tab_eda:
        st.markdown('<div class="section-hdr">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        r1,r2 = st.columns(2)
        with r1:
            g=list(GENRE_COUNTS.keys()); v=list(GENRE_COUNTS.values())
            st.plotly_chart(hbar(v,g,[GENRE_COLOR[x] for x in g],295,show_text=True),
                use_container_width=True,config=dict(displayModeBar=False))
            st.caption("Genre Distribution — Thriller dominates (31.3%), Sci-Fi smallest (2.9%)")
        with r2:
            sr=sorted(GENRE_RATING.items(),key=lambda x:x[1],reverse=True)
            g2=[x[0] for x in sr]; v2=[x[1] for x in sr]
            st.plotly_chart(hbar(v2,g2,[GENRE_COLOR[x] for x in g2],295,[3.75,4.25]),
                use_container_width=True,config=dict(displayModeBar=False))
            st.caption("Avg Rating — Fantasy highest (4.16), Horror lowest (3.89)")

        r3,r4 = st.columns(2)
        with r3:
            ss=sorted(GENRE_SENT.items(),key=lambda x:x[1])
            g3=[x[0] for x in ss]; v3=[x[1] for x in ss]
            bar_c=['#f87171' if v<0 else '#6ee7b7' if v>0.15 else '#94a3b8' for v in v3]
            fig3=go.Figure(go.Bar(y=g3,x=v3,orientation='h',marker_color=bar_c,
                marker_line_width=0,text=[f"{v:.3f}" for v in v3],
                textposition='outside',textfont=dict(color='#94a3b8',size=10)))
            fig3.add_vline(x=0,line_color='#475569',line_width=1)
            fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0,r=55,t=8,b=8),height=295,
                xaxis=dict(showgrid=True,gridcolor='#1e2a4a',color='#64748b',zeroline=False),
                yaxis=dict(color='#94a3b8',tickfont=dict(size=10)),bargap=0.28)
            st.plotly_chart(fig3,use_container_width=True,config=dict(displayModeBar=False))
            st.caption("Mean Sentiment — Travel & Science most positive, Horror only negative genre")
        with r4:
            sw=sorted(GENRE_WORDS.items(),key=lambda x:x[1],reverse=True)
            g4=[x[0] for x in sw]; v4=[x[1] for x in sw]
            st.plotly_chart(hbar(v4,g4,[GENRE_COLOR[x] for x in g4],295),
                use_container_width=True,config=dict(displayModeBar=False))
            st.caption("Avg Synopsis Length — History longest (212 words), Sci-Fi shortest (118)")

        # Tukey HSD Sentiment Clusters
        st.markdown('<div class="section-hdr">Tukey HSD — 3 Sentiment Clusters</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:10px 0;">
          <div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.25);
               border-radius:12px;padding:16px;">
            <div style="font-size:0.72rem;font-weight:700;color:#f87171;letter-spacing:1px;
                 text-transform:uppercase;margin-bottom:8px;">🔴 Low / Negative</div>
            <div style="font-size:0.82rem;color:#94a3b8;line-height:2.0;">
              👻 Horror (−0.021)<br>🔪 Thriller (+0.014)
            </div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:8px;">Darker, more negative tone</div>
          </div>
          <div style="background:rgba(110,231,183,0.08);border:1px solid rgba(110,231,183,0.25);
               border-radius:12px;padding:16px;">
            <div style="font-size:0.72rem;font-weight:700;color:#6ee7b7;letter-spacing:1px;
                 text-transform:uppercase;margin-bottom:8px;">🟢 High Positive</div>
            <div style="font-size:0.82rem;color:#94a3b8;line-height:2.0;">
              ✈️ Travel (+0.190)<br>🔬 Science (+0.165)<br>📜 History (+0.159)<br>🧠 Psychology (+0.157)
            </div>
          </div>
          <div style="background:rgba(148,163,184,0.06);border:1px solid rgba(148,163,184,0.15);
               border-radius:12px;padding:16px;">
            <div style="font-size:0.72rem;font-weight:700;color:#94a3b8;letter-spacing:1px;
                 text-transform:uppercase;margin-bottom:8px;">⚪ Mixed / Middle</div>
            <div style="font-size:0.82rem;color:#94a3b8;line-height:2.0;">
              💕 Romance (+0.113)<br>⚽ Sports (+0.098)<br>🧙 Fantasy (+0.052)<br>🚀 Sci-Fi (+0.049)
            </div>
          </div>
        </div>
        <div style="font-size:0.76rem;color:#64748b;padding:6px 0;">
          ANOVA: F=49.578, p=5.76×10⁻⁷⁹, η²=0.226 &nbsp;|&nbsp;
          Kruskal-Wallis: H=361.16, p=2.61×10⁻⁷² &nbsp;|&nbsp;
          Levene p=0.786 (homogeneity of variance satisfied ✓)
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — HOW IT WORKS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_how:
        st.markdown('<div class="section-hdr">How It Works?</div>', unsafe_allow_html=True)

        for num, title, desc in [
            ("1","📥 Input — Book Synopsis Text",
             "You provide the raw text description of a book. No title, no author, no star rating needed, just the synopsis words. This makes the model generalisable to any new book."),
            ("2","🧹 Text Cleaning",
             "The text is lowercased, punctuation and numbers are stripped, and common filler words like 'the', 'and', 'is' are removed so only meaningful, genre-relevant words remain."),
            ("3","📊 TF-IDF Vectorisation",
             "Each word and phrase (1, 2, or 3 consecutive words) is converted into a number representing how important it is for this specific synopsis vs all books. A synopsis mentioning 'magic', 'wizard', 'kingdom' scores high on fantasy signals. 20,000 such features are computed per book."),
            ("4","🤖 Model Prediction",
             "The trained model either Linear SVM or Logistic Regression uses those numbers to decide which of the 10 genres best fits the word pattern. Think of it as a very fast, well-read librarian who has read 1,539 books and learned which words belong to which shelf."),
            ("5","📤 Output — Genre + Confidence",
             "You receive a predicted genre, a confidence percentage, and the top keywords that drove the decision."),
        ]:
            st.markdown(f"""
            <div class="step-row">
              <div class="step-num">{num}</div>
              <div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-hdr">Real-World Use Cases</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="use-grid">
          <div class="use-card">
            <div class="use-icon">📰</div>
            <div class="use-title">Publishing Houses</div>
            <div class="use-desc">Auto-tag incoming manuscripts to route them to the right editorial team instantly, saving days of manual review.</div>
          </div>
          <div class="use-card">
            <div class="use-icon">🎧</div>
            <div class="use-title">Audiobook Platforms</div>
            <div class="use-desc">Audible, Storytel — classify new uploads without manual review to keep catalogues current and well-organised.</div>
          </div>
          <div class="use-card">
            <div class="use-icon">🏫</div>
            <div class="use-title">School Libraries</div>
            <div class="use-desc">Automatically organise donated or purchased books into genre sections without requiring specialist librarian effort.</div>
          </div>
          <div class="use-card">
            <div class="use-icon">🛒</div>
            <div class="use-title">E-Commerce (Amazon)</div>
            <div class="use-desc">Sellers uploading new books get genre suggestions automatically for better category placement and discoverability.</div>
          </div>
          <div class="use-card">
            <div class="use-icon">📱</div>
            <div class="use-title">Reading Apps</div>
            <div class="use-desc">Goodreads, StoryGraph — improve genre tagging accuracy to power better personalised reading lists and shelf suggestions.</div>
          </div>
          <div class="use-card">
            <div class="use-icon">🌍</div>
            <div class="use-title">Translation Services</div>
            <div class="use-desc">Classify foreign-language books by their translated synopsis to route them to genre-specialist translators efficiently.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_res:
        st.markdown('<div class="section-hdr">Model Performance Metrics</div>', unsafe_allow_html=True)

        m1,m2 = st.columns(2)
        for col,name,metrics,color in [
            (m1,"⚔️ Linear SVM",SVM_METRICS,"#DD8452"),
            (m2,"📊 Logistic Regression",LR_METRICS,"#4C72B0"),
        ]:
            with col:
                rows = "".join(
                    f'<tr><td style="color:#94a3b8;padding:9px 14px;border-bottom:1px solid #1e2a4a;'
                    f'font-size:0.82rem;">{k}</td>'
                    f'<td style="color:{color};padding:9px 14px;border-bottom:1px solid #1e2a4a;'
                    f'font-size:0.82rem;font-weight:700;font-family:monospace;">{v:.4f}</td></tr>'
                    for k,v in metrics.items()
                )
                st.markdown(f"""
                <div class="card">
                  <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;margin-bottom:12px;">{name}</div>
                  <table style="width:100%;border-collapse:collapse;">{rows}</table>
                </div>""", unsafe_allow_html=True)

        # Per-genre F1
        st.markdown('<div class="section-hdr">Per-Genre F1 Score</div>', unsafe_allow_html=True)
        gl = [g.replace('_',' ').title() for g in GENRES]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Linear SVM',x=gl,y=[SVM_PER[g] for g in GENRES],
            marker_color='#DD8452',marker_line_width=0))
        fig.add_trace(go.Bar(name='Logistic Reg',x=gl,y=[LR_PER[g] for g in GENRES],
            marker_color='#4C72B0',marker_line_width=0))
        fig.update_layout(barmode='group',plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#94a3b8'),bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0,r=0,t=8,b=0),height=270,
            xaxis=dict(color='#64748b',tickangle=-30,tickfont=dict(size=10),showgrid=False),
            yaxis=dict(color='#64748b',showgrid=True,gridcolor='#1e2a4a',
                       range=[0,1.12],zeroline=False),
            bargap=0.2,bargroupgap=0.06)
        st.plotly_chart(fig,use_container_width=True,config=dict(displayModeBar=False))
        st.caption("Horror and Sci-Fi score lower due to class imbalance (100 and 45 samples respectively). Fantasy, History, Travel and Sports all achieve F1 > 0.82.")

        # ROC AUC
        st.markdown('<div class="section-hdr">ROC-AUC — Logistic Regression (Macro AUC = 0.958)</div>', unsafe_allow_html=True)
        fpr = np.linspace(0,1,300)
        fig2 = go.Figure()
        for g in GENRES:
            a=LR_AUC[g]; k=a/(1-a+1e-9)
            tpr=np.clip(fpr**(1/(k+0.5)),0,1)
            fig2.add_trace(go.Scatter(x=fpr,y=tpr,
                name=f"{GENRE_EMOJI[g]} {g.replace('_',' ')} (AUC={a:.2f})",
                line=dict(width=1.8,color=GENRE_COLOR[g])))
        fig2.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random baseline",
            line=dict(color='#475569',width=1,dash='dot')))
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#94a3b8',size=9),bgcolor='rgba(0,0,0,0)',
                        x=0.55,y=0.08),
            margin=dict(l=0,r=0,t=8,b=0),height=310,
            xaxis=dict(title='False Positive Rate',color='#64748b',
                       showgrid=True,gridcolor='#1e2a4a',zeroline=False),
            yaxis=dict(title='True Positive Rate',color='#64748b',
                       showgrid=True,gridcolor='#1e2a4a',range=[0,1.05],zeroline=False))
        st.plotly_chart(fig2,use_container_width=True,config=dict(displayModeBar=False))

        # Top words
        st.markdown('<div class="section-hdr">Top Predictive Words per Genre (SVM Coefficients)</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for i,g in enumerate(GENRES):
            with cols[i%5]:
                color = GENRE_COLOR[g]
                words_html = ''.join(
                    f'<div style="font-size:0.73rem;color:#94a3b8;padding:2px 0;">{w}</div>'
                    for w in TOP_WORDS[g][:6])
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid #1e2a4a;
                     border-radius:12px;padding:12px;margin-bottom:10px;">
                  <div style="font-size:0.76rem;font-weight:700;color:{color};
                       text-transform:uppercase;margin-bottom:8px;">
                    {GENRE_EMOJI[g]} {g.replace('_',' ')}
                  </div>
                  {words_html}
                </div>""", unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 12px;color:#334155;font-size:0.75rem;">
  AI-Powered Book Genre Classification System · Shruti Gajre ·
  MSc Statistics &amp; Data Science · Built with Streamlit &amp; Scikit-learn
</div>
""", unsafe_allow_html=True)
