# -*- coding: utf-8 -*-
"""
Aksara Match — Smart Book Recommendations (Dark Library Edition)
Hybrid recommender: Collaborative Filtering (SVD) + Content-Based (TF-IDF)

Jalankan:  streamlit run app.py
"""
import html

import numpy as np
import pandas as pd
import streamlit as st

import recsys

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aksara Match — Book Recommendations",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark Library Theme (palet & gaya dari referensi CineMatch)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Root Variables ── */
:root {
    --gold:       #E8B84B;
    --gold-dim:   #a07c28;
    --red:        #C0392B;
    --bg:         #0a0a0f;
    --bg2:        #111118;
    --bg3:        #1a1a26;
    --card:       #16161f;
    --card-hover: #1e1e2e;
    --border:     rgba(232,184,75,0.15);
    --text:       #e8e6e0;
    --text-dim:   #888899;
    --font-head:  'Playfair Display', Georgia, serif;
    --font-body:  'DM Sans', sans-serif;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background-color: var(--bg); }

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
/* Preserve sidebar toggle button inside header */
[data-testid="stHeader"] button,
[data-testid="collapsedControl"],
button[kind="header"],
.st-emotion-cache-dvne4q,
button[title="Open sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Close sidebar"] {
    visibility: visible !important;
    pointer-events: auto !important;
}
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--font-body);
    font-size: 0.9rem;
    color: var(--text-dim);
    cursor: pointer;
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--gold); }
[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div + label { color: var(--gold); }

/* ── Headings ── */
h1, h2, h3 {
    font-family: var(--font-head);
    color: var(--text);
}

/* ── Gold Divider ── */
hr { border-color: var(--border) !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}
[data-testid="metric-container"] label {
    color: var(--text-dim) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-family: var(--font-head);
    font-size: 2.2rem !important;
}

/* ── Text Input ── */
[data-testid="stTextInput"] input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-body);
    font-size: 1rem;
    padding: 0.65rem 1rem !important;
    transition: border 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(232,184,75,0.12) !important;
}

/* ── Select Box ── */
[data-baseweb="select"] > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--gold) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.04em;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #f5cc6a !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(232,184,75,0.25) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--gold), #d4a035) !important;
}

/* ── Progress Bar ── */
[data-testid="stProgressBar"] > div { background: var(--bg3) !important; border-radius: 99px; }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, var(--gold), #f5cc6a) !important; border-radius: 99px; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; gap: 0.5rem; }
[data-baseweb="tab"] {
    font-family: var(--font-body) !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 1.2rem !important;
    border-radius: 6px 6px 0 0;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: rgba(232,184,75,0.06) !important;
}

/* ── Info / Warning Boxes ── */
[data-testid="stAlert"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p { color: var(--gold) !important; }

/* ── Multiselect ── */
[data-baseweb="tag"] {
    background: rgba(232,184,75,0.15) !important;
    color: var(--gold) !important;
    border-radius: 6px !important;
}

/* ── Matplotlib / Charts ── */
.stPlotlyChart, .stPyplot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# EXTRA CSS (kartu buku, chip, dll.)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.am-card{background:#16161f;border:1px solid rgba(232,184,75,0.15);border-radius:12px;
         overflow:hidden;margin-bottom:0.5rem;transition:all .25s;}
.am-card:hover{border-color:rgba(232,184,75,0.45);transform:translateY(-2px);}
.am-title{font-family:'Playfair Display',serif;font-weight:700;color:#e8e6e0;line-height:1.35;}
.am-sub{color:#888899;font-size:0.8rem;margin-top:2px;}
.am-label{font-size:0.68rem;color:#888899;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.25rem;}
.am-chip{display:inline-block;background:rgba(232,184,75,0.1);border:1px solid rgba(232,184,75,0.3);
         color:#E8B84B;padding:2px 10px;border-radius:20px;font-size:0.72rem;margin:0 6px 6px 0;}
.am-reason{color:#888899;font-size:0.78rem;font-style:italic;margin-top:0.5rem;line-height:1.4;}
.am-section{font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#888899;margin:1rem 0 0.4rem;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Opening the library…")
def load_all():
    art = recsys.load_artifacts("artifacts/model.joblib")
    it = art["items"].copy()
    it["decade"] = np.where(it["year"].notna(),
                            (it["year"] // 10 * 10).fillna(0).astype(int).astype(str) + "s", "Unknown")
    art["items"] = it
    lookup = it.set_index("isbn", drop=False)
    top_auth = it.groupby("author")["n_inter"].sum().nlargest(300).index.tolist()
    top_pub = it.groupby("publisher")["n_inter"].sum().nlargest(150).index.tolist()
    decades = sorted(d for d in it["decade"].unique() if d != "Unknown")
    n_ex = it["n_explicit"].sum()
    avg_all = float((it["avg_rating"].fillna(0) * it["n_explicit"]).sum() / n_ex) if n_ex else 0.0
    return art, lookup, top_auth, top_pub, decades, avg_all


try:
    art, LOOKUP, TOP_AUTHORS, TOP_PUBS, DECADES, AVG_ALL = load_all()
except FileNotFoundError:
    st.error("File `artifacts/model.joblib` not found. Run `python train_model.py` first "
             "(or place the model file produced by the Kaggle notebook in `artifacts/`).")
    st.stop()

items, eda, metrics, TOP_K = art["items"], art["eda"], art["metrics"], art["top_k"]
GOLD, BG, CARD = "#E8B84B", "#111118", "#16161f"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def esc(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return html.escape(str(x))


def H(s):
    """Rapatkan HTML (hilangkan indentasi/baris kosong) agar tidak dibaca sebagai code-block markdown."""
    return "".join(line.strip() for line in s.strip().splitlines())


def year_txt(y):
    return str(int(y)) if pd.notna(y) else "—"


def rating_txt(r):
    return f"★ {r:.1f}" if pd.notna(r) else "★ —"


def stars10(r):
    n = int(round(r / 2)) if pd.notna(r) else 0
    return "★" * n + "☆" * (5 - n)


def chips(row):
    parts = [row.get("author"), row.get("publisher"), row.get("decade")]
    parts = [p for p in parts if isinstance(p, str) and p and p != "Unknown"]
    return "".join(f'<span class="am-chip">{esc(p)}</span>' for p in parts[:3])


def cover_html(url, h=300):
    if isinstance(url, str) and url.startswith("http"):
        url = url.replace("http://", "https://")
        return (f'<img src="{esc(url)}" style="width:100%;height:{h}px;object-fit:contain;'
                f'background:#0d0d18;display:block;">')
    return (f'<div style="width:100%;height:{h}px;background:#0d0d18;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:center;gap:.5rem;">'
            f'<div style="font-size:2.5rem;opacity:.25;">📖</div>'
            f'<div style="font-size:.8rem;color:#555566;">No Cover</div></div>')


def open_detail(isbn):
    st.session_state["detail_book"] = isbn


def detail_button(isbn, key):
    if st.button("View Details", key=key, use_container_width=True):
        open_detail(isbn)
        st.rerun()


def rec_card(row, key):
    score = float(row["score"])
    pct = max(0.0, min(score * 100, 100.0))
    reason = esc(row.get("reason", ""))
    reason_html = f'<div class="am-reason">{reason}</div>' if reason and reason != "-" else ""
    st.markdown(H(f"""
    <div class="am-card">
        {cover_html(row['img_m'], 300)}
        <div style="padding:1rem 1.1rem 0.9rem;">
            <div class="am-title" style="font-size:1rem;">{esc(row['title'])}</div>
            <div class="am-sub">{esc(row['author'])} · {year_txt(row['year'])}</div>
            <div style="margin-top:.6rem;">{chips(row)}</div>
            <div class="am-label" style="margin-top:.4rem;">Match Score</div>
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="flex:1;background:#1a1a26;border-radius:99px;height:6px;overflow:hidden;">
                    <div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,#E8B84B,#f5cc6a);border-radius:99px;"></div>
                </div>
                <span style="color:#E8B84B;font-size:.85rem;font-weight:600;">{score:.2f}</span>
            </div>
            <div style="color:#555566;font-size:.72rem;margin-top:.35rem;">
                Readers' taste {row['s_cf']:.2f} · Content {row['s_content']:.2f}</div>
            {reason_html}
        </div>
    </div>
    """), unsafe_allow_html=True)
    detail_button(row["isbn"], key)


def mini_card(row, key, h=190, show_rating=True):
    r = row.get("avg_rating")
    rating_html = (f'<div style="color:#E8B84B;font-size:1rem;font-weight:700;margin-top:.4rem;">'
                   f'{rating_txt(r)}</div>') if show_rating else ""
    st.markdown(H(f"""
    <div class="am-card" style="text-align:center;">
        {cover_html(row['img_m'], h)}
        <div style="padding:.8rem .7rem .7rem;">
            <div class="am-title" style="font-size:.82rem;min-height:40px;">{esc(str(row['title'])[:42])}{'…' if len(str(row['title'])) > 42 else ''}</div>
            <div class="am-sub">{esc(str(row['author'])[:28])}</div>
            {rating_html}
            <div style="color:#555566;font-size:.72rem;">{int(row['n_inter']):,} readers</div>
        </div>
    </div>
    """), unsafe_allow_html=True)
    detail_button(row["isbn"], key)


def browse_card(row, key):
    r = row.get("avg_rating")
    st.markdown(H(f"""
    <div class="am-card">
        <div style="display:flex;">
            <div style="width:38%;flex-shrink:0;">{cover_html(row['img_m'], 200)}</div>
            <div style="padding:1rem 1rem .8rem;flex:1;">
                <div class="am-title" style="font-size:.95rem;">{esc(str(row['title'])[:70])}</div>
                <div class="am-sub">{esc(row['author'])}</div>
                <div class="am-sub">{year_txt(row['year'])} · {esc(row['publisher'])}</div>
                <div style="color:#E8B84B;font-weight:700;margin-top:.6rem;">{rating_txt(r)}</div>
                <div style="color:#555566;font-size:.72rem;">{int(row['n_inter']):,} readers</div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)
    detail_button(row["isbn"], key)


def render_recs(recs, eyebrow, title, prefix):
    if recs is None or len(recs) == 0:
        st.warning("No recommendations found. Try widening the year range or another book.")
        return
    st.markdown(H(f"""
    <div style="margin:2rem 0 1.5rem;">
        <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;color:#E8B84B;margin-bottom:.4rem;">{esc(eyebrow)}</div>
        <h2 style="font-family:'Playfair Display',serif;font-size:2rem;margin:0;">{esc(title)}</h2>
    </div>
    <hr style="border-color:rgba(232,184,75,0.12);margin-bottom:1.5rem;">
    """), unsafe_allow_html=True)
    for start in range(0, len(recs), 2):
        cols = st.columns(2, gap="medium")
        for ci, (_, row) in enumerate(recs.iloc[start:start + 2].iterrows()):
            with cols[ci]:
                rec_card(row, f"{prefix}_{start}_{ci}_{row['isbn']}")


def page_title(title, subtitle):
    st.markdown(H(f"""
    <h1 style="font-family:'Playfair Display',serif;font-size:2.4rem;margin-bottom:.3rem;">{title}</h1>
    <p style="color:#888899;margin-bottom:1.5rem;">{subtitle}</p>
    """), unsafe_allow_html=True)


def sidebar_label(text):
    st.sidebar.markdown(H(f"""
    <div style="font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;color:#888899;
                margin-bottom:.5rem;margin-top:1.2rem;">{text}</div>
    """), unsafe_allow_html=True)


# ── pengaturan rekomendasi (dengan tombol reset) ──
SETTING_KEYS = ["k", "max_auth", "yr_rng", "use_custom", "w_cf"]


def reset_settings():
    for key in SETTING_KEYS:
        st.session_state.pop(key, None)


def sidebar_settings():
    sidebar_label("Recommendation Settings")
    st.sidebar.button("🔄 Reset settings", on_click=reset_settings, use_container_width=True)
    k = st.sidebar.slider("Books to recommend", 5, 20, 10, key="k")
    max_auth = st.sidebar.slider("Max books per author", 1, 5, 2, key="max_auth")
    yv = items["year"].dropna()
    lo, hi = int(yv.min()), int(yv.max())
    yr = st.sidebar.slider("Publication year", lo, hi, (lo, hi), key="yr_rng")
    sidebar_label("Hybrid Weights")
    bw = art["best_w"]
    if st.sidebar.checkbox("Set weights manually", value=False, key="use_custom"):
        w_cf = st.sidebar.slider("Readers' taste (collaborative)", 0.0, 1.0, float(bw[0]), 0.05, key="w_cf")
        weights = (w_cf, round(1 - w_cf, 2))
        st.sidebar.caption(f"Content-based weight = {weights[1]}")
    else:
        weights = bw
        st.sidebar.info(f"Best weights from evaluation: collaborative {bw[0]} · content {bw[1]}")
    return dict(k=k, weights=weights, year_min=yr[0], year_max=yr[1], max_per_author=max_auth)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem;text-align:center;">
        <div style="font-size:2.4rem;margin-bottom:0.3rem;">📖</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:900;
                    background:linear-gradient(135deg,#E8B84B,#f5cc6a);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Aksara Match
        </div>
        <div style="font-size:0.72rem;color:#888899;letter-spacing:0.12em;text-transform:uppercase;margin-top:2px;">
            Smart Book Recommendations
        </div>
    </div>
    <hr style="border-color:rgba(232,184,75,0.15);margin:0 0 1.2rem;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Menu",
        ["🏠  Home", "📊  Analytics", "🎯  Recommend", "📚  Browse Library"],
        label_visibility="collapsed",
    )

    if st.session_state.get("detail_book"):
        st.markdown("<hr style='border-color:rgba(232,184,75,0.1);margin:1rem 0;'>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            st.session_state["detail_book"] = None
            st.rerun()

    st.markdown("<hr style='border-color:rgba(232,184,75,0.1);margin:1.5rem 0 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: BOOK DETAIL
# ─────────────────────────────────────────────────────────────
if st.session_state.get("detail_book"):
    isbn = st.session_state["detail_book"]
    if isbn not in LOOKUP.index:
        st.session_state["detail_book"] = None
        st.rerun()
    b = LOOKUP.loc[isbn]
    col_cover, col_info = st.columns([1, 2], gap="large")

    with col_cover:
        url = b["img_m"] if isinstance(b["img_m"], str) else None
        if url:
            url = url.replace("http://", "https://")
            st.markdown(f'<img src="{esc(url)}" style="width:100%;border-radius:14px;'
                        f'box-shadow:0 8px 40px rgba(0,0,0,0.6);display:block;background:#0d0d18;">',
                        unsafe_allow_html=True)
        else:
            st.markdown(H("""
            <div style="width:100%;height:380px;background:#16161f;border:1px solid rgba(232,184,75,0.15);
                        border-radius:14px;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;gap:.5rem;">
                <div style="font-size:3rem;opacity:.2;">📖</div>
                <div style="font-size:.8rem;color:#888899;">No Cover Available</div>
            </div>"""), unsafe_allow_html=True)

    with col_info:
        st.markdown(H(f"""
        <div style="margin-bottom:.6rem;">
            <h1 style="font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;margin:0 0 .3rem;line-height:1.15;">{esc(b['title'])}</h1>
            <div style="color:#E8B84B;font-size:1.05rem;margin-bottom:.3rem;">by {esc(b['author'])}</div>
            <div style="color:#888899;font-size:.9rem;">{year_txt(b['year'])} · {esc(b['publisher'])}</div>
        </div>"""), unsafe_allow_html=True)
        st.markdown(f'<div style="margin:.8rem 0 1rem;">{chips(b)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="am-label" style="margin-bottom:.6rem;">Rating</div>', unsafe_allow_html=True)
        r = b["avg_rating"]
        if pd.notna(r):
            st.markdown(H(f"""
            <div style="background:#16161f;border:1px solid rgba(232,184,75,0.15);border-radius:12px;
                        padding:1rem 1.2rem;display:inline-block;min-width:200px;">
                <div class="am-label">Reader Rating (Dataset)</div>
                <div style="color:#E8B84B;font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;">
                    {r:.2f}<span style="font-size:1rem;color:#555566;">/10</span></div>
                <div style="color:#E8B84B;font-size:.9rem;margin-top:.2rem;">{stars10(r)}</div>
                <div style="color:#555566;font-size:.72rem;margin-top:.2rem;">from {int(b['n_explicit']):,} explicit ratings</div>
            </div>"""), unsafe_allow_html=True)
        else:
            st.markdown(H("""
            <div style="background:#16161f;border:1px solid rgba(232,184,75,0.15);border-radius:12px;
                        padding:1rem 1.2rem;display:inline-block;min-width:200px;">
                <div class="am-label">Reader Rating (Dataset)</div>
                <div style="color:#555566;font-size:1.5rem;">—</div>
                <div style="color:#555566;font-size:.72rem;">no explicit ratings yet</div>
            </div>"""), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("👥 Readers", f"{int(b['n_inter']):,}")
        m2.metric("⚖️ Weighted Rating", f"{b['bayes_rating']:.2f}")
        m3.metric("🔖 ISBN", str(b["isbn"]))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<h2 style="font-family:'Playfair Display',serif;font-size:1.6rem;margin-bottom:1.2rem;">
        ✨ More Like This</h2>""", unsafe_allow_html=True)
    sim = recsys.similar_books(art, isbn, k=6)
    if len(sim):
        sim = sim.merge(items[["isbn", "img_m", "avg_rating", "n_inter"]], on="isbn", how="left", suffixes=("", "_x"))
        cols = st.columns(6)
        for i, (_, row) in enumerate(sim.iterrows()):
            with cols[i]:
                mini_card(row, f"sim_{isbn}_{i}", h=170)
    st.stop()


# ─────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown("""
    <div style="position:relative;padding:4rem 2rem 3.5rem;margin:-2rem -2.5rem 2rem;
                background:linear-gradient(135deg,#0a0a0f 0%,#12121e 50%,#0f0f18 100%);
                border-bottom:1px solid rgba(232,184,75,0.12);overflow:hidden;">
        <div style="position:absolute;top:-60px;right:-60px;width:320px;height:320px;
                    background:radial-gradient(circle,rgba(232,184,75,0.06) 0%,transparent 70%);
                    border-radius:50%;pointer-events:none;"></div>
        <div style="position:absolute;bottom:-80px;left:10%;width:500px;height:200px;
                    background:radial-gradient(ellipse,rgba(192,57,43,0.05) 0%,transparent 70%);
                    pointer-events:none;"></div>
        <div style="position:relative;max-width:800px;">
            <h1 style="font-family:'Playfair Display',serif;font-size:clamp(2.2rem,4vw,3.6rem);
                       font-weight:900;line-height:1.1;margin:0 0 1rem;
                       background:linear-gradient(135deg,#e8e6e0,#E8B84B 60%,#f5cc6a);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                Discover Your Next<br>Favorite Book
            </h1>
            <p style="color:#888899;font-size:1.05rem;max-width:540px;line-height:1.7;margin:0;">
                Aksara Match blends collaborative filtering with content-based intelligence
                to surface books you'll love — before you even know to look for them.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Books in Model", f"{eda['model_items']:,}")
    c2.metric("👥 Readers", f"{eda['model_users']:,}")
    c3.metric("🔗 Interactions", f"{eda['model_inter']:,}")
    c4.metric("⭐ Avg Rating", f"{AVG_ALL:.2f} / 10")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<h2 style="font-family:'Playfair Display',serif;font-size:1.6rem;margin-bottom:1.5rem;">
        How It Works</h2>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, icon, ttl, desc in [
        (col1, "🔍", "Pick a Book", "Search any title or author. Content-based TF-IDF understands titles, authors and publishers."),
        (col2, "🤝", "Find Your Tribe", "A latent-factor model (SVD) finds readers with similar taste who read the same books."),
        (col3, "✨", "Get Matched", "Both signals are blended into one hybrid score, so you get books that fit — not just the obvious picks."),
    ]:
        col.markdown(H(f"""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.5rem;height:100%;">
            <div style="font-size:2rem;margin-bottom:.8rem;">{icon}</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:#e8e6e0;margin-bottom:.5rem;">{ttl}</div>
            <div style="color:#888899;font-size:.88rem;line-height:1.6;">{desc}</div>
        </div>"""), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<h2 style="font-family:'Playfair Display',serif;font-size:1.6rem;margin-bottom:1.2rem;">
        🔥 Top Rated Books in Dataset</h2>""", unsafe_allow_html=True)

    pool = items[items["n_explicit"] >= 30]
    if len(pool) < 6:
        pool = items[items["n_explicit"] >= 10]
    top_books = pool.sort_values("bayes_rating", ascending=False).head(6)
    cols = st.columns(6)
    for i, (_, row) in enumerate(top_books.iterrows()):
        with cols[i]:
            mini_card(row, f"home_{i}_{row['isbn']}")


# ─────────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────
elif page == "📊  Analytics":
    page_title("Dataset Analytics", "Deep-dive into the reading landscape behind the recommendations.")

    # ── chart HTML/CSS murni (tanpa library tambahan) ──
    def chart_card(title, inner):
        st.markdown(H(f"""
        <div style="background:#111118;border:1px solid rgba(232,184,75,0.12);border-radius:14px;
                    padding:1.2rem 1.4rem;margin-bottom:1rem;">
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;
                        color:#e8e6e0;margin-bottom:.9rem;">{esc(title)}</div>
            {inner}
        </div>"""), unsafe_allow_html=True)

    def hbars(series, title, color=GOLD, highlight=None, fmt="{:,.0f}", label_w=150):
        mx = float(max(series.values)) if len(series) else 1.0
        rows = ""
        for lab, val in series.items():
            w = 100 * float(val) / mx if mx > 0 else 0
            c = color if highlight is None else (GOLD if lab == highlight else "#2a2a3e")
            rows += (f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
                     f'<div style="width:{label_w}px;text-align:right;color:#e8e6e0;font-size:.8rem;'
                     f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{esc(lab)}">{esc(lab)}</div>'
                     f'<div style="flex:1;background:#1a1a26;border-radius:6px;height:18px;overflow:hidden;">'
                     f'<div style="width:{w:.1f}%;height:100%;background:{c};border-radius:6px;"></div></div>'
                     f'<div style="width:72px;color:#888899;font-size:.78rem;">{fmt.format(float(val))}</div></div>')
        chart_card(title, rows)

    def vbars(labels, values, title, highlight=None, height=200, note=""):
        vals = [float(v) for v in values]
        mx = max(vals) if vals else 1.0
        step = max(1, len(labels) // 12)
        bars, labs = "", ""
        for i, (lab, v) in enumerate(zip(labels, vals)):
            hgt = 100 * v / mx if mx > 0 else 0
            c = GOLD if (highlight is None or str(lab) in highlight) else "#2a2a3e"
            bars += (f'<div title="{esc(lab)}: {v:,.0f}" style="flex:1;height:{hgt:.1f}%;min-height:1px;'
                     f'background:{c};border-radius:3px 3px 0 0;"></div>')
            txt = esc(lab) if i % step == 0 else ""
            labs += (f'<div style="flex:1;display:flex;justify-content:center;">'
                     f'<span style="color:#888899;font-size:.66rem;white-space:nowrap;">{txt}</span></div>')
        inner = (f'<div style="display:flex;align-items:flex-end;gap:2px;height:{height}px;'
                 f'border-bottom:1px solid #1e1e2e;">{bars}</div>'
                 f'<div style="display:flex;gap:2px;margin-top:4px;">{labs}</div>')
        if note:
            inner += f'<div style="color:#555566;font-size:.72rem;margin-top:.6rem;">{esc(note)}</div>'
        chart_card(title, inner)

    t1, t2, t3, t4 = st.tabs(["⭐ Rating Distribution", "🌍 Readers & Authors",
                              "📅 Publication Years", "🧪 Model Evaluation"])

    with t1:
        rd = pd.Series(eda["rating_dist"]).sort_index()
        vbars([str(int(i)) for i in rd.index], rd.values, "Rating Distribution (0 = read, not rated)",
              highlight={"0"}, height=240,
              note=f"{eda['pct_implicit']*100:.0f}% of all interactions are implicit (rating 0). "
                   "They are kept as positive signals in the model.")

    with t2:
        a_col, b_col = st.columns(2)
        with a_col:
            tc = pd.Series(eda["top_countries"]).sort_values(ascending=False)
            hbars(tc, "Top Countries by Readers", color=GOLD, label_w=120)
        with b_col:
            ta = pd.Series(eda["top_authors"]).sort_values(ascending=False)
            hbars(ta, "Most Read Authors", color="#C0392B", label_w=150)
        ah = pd.Series(eda["age_hist"]).sort_index()
        ah = ah[(ah.index >= 5) & (ah.index <= 90)]
        vbars([str(int(i)) for i in ah.index], ah.values, "Reader Age Distribution (where available)",
              highlight={str(int(ah.idxmax()))}, height=170)

    with t3:
        yh = pd.Series(eda["year_hist"]).sort_index()
        yh = yh[yh.index >= 1950]
        vbars([str(int(i)) for i in yh.index], yh.values, "Books by Publication Year", height=240)

    with t4:
        st.caption(f"Leave-one-out: one liked book per reader is hidden, and the model must find it "
                   f"among {eda['model_items']:,} books. Higher is better.")
        hit, ndcg = f"HitRate@{TOP_K}", f"NDCG@{TOP_K}"
        st.dataframe(metrics.style.format({hit: "{:.2%}", ndcg: "{:.4f}", "MRR": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        m = metrics.sort_values(hit, ascending=False)
        best_model = m.iloc[0]["model"]
        hbars(pd.Series((m[hit] * 100).values, index=m["model"].values),
              f"HitRate@{TOP_K} per Model", highlight=best_model, fmt="{:.2f}%", label_w=190)


# ─────────────────────────────────────────────────────────────
# PAGE: RECOMMEND
# ─────────────────────────────────────────────────────────────
elif page == "🎯  Recommend":
    page_title("Find Your Next Book", "Tell us a book you love — or a reader profile — and we'll find what to read next.")
    settings = sidebar_settings()

    tab_book, tab_reader = st.tabs(["📖 By a Book I Like", "👤 By Reader ID"])

    # ---------- by book ----------
    with tab_book:
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            q = st.text_input("Search", placeholder="e.g. The Firm, Harry Potter, Stephen King…",
                              label_visibility="collapsed", key="rec_query")
        with col_btn:
            st.button("Search", use_container_width=True)

        if q and len(q.strip()) >= 2:
            cands = recsys.search_books(art, q, limit=30)
            if len(cands) == 0:
                st.warning(f"No results for **{q}**. Try a different spelling.")
            else:
                st.markdown('<div class="am-section">Did you mean?</div>', unsafe_allow_html=True)
                options = {f"{r.title[:80]} — {r.author}": r.isbn for r in cands.itertuples()}
                chosen = st.selectbox("Did you mean", list(options), label_visibility="collapsed")
                your_rating = st.slider("How much did you like it?", 5, 10, 9)
                st.markdown("<div style='margin-top:.6rem;'></div>", unsafe_allow_html=True)
                if st.button("✨ Get Recommendations", type="primary"):
                    isbn = options[chosen]
                    with st.spinner("Curating your personal reading list…"):
                        x = recsys.vector_from_isbns(art, {isbn: your_rating})
                        recs = recsys.recommend(art, x, **settings)
                    st.session_state["last_recs"] = recs.to_dict("records") if len(recs) else []
                    st.session_state["last_recs_title"] = chosen.split(" — ")[0]

        if "last_recs" in st.session_state:
            render_recs(pd.DataFrame(st.session_state["last_recs"]),
                        "Because you like", st.session_state.get("last_recs_title", ""), "rb")

    # ---------- by reader ----------
    with tab_reader:
        users = art["users"]
        c1, c2, c3 = st.columns([2, 1, 1])
        min_i = c3.number_input("Min. books read", 5, 500, 10)
        pool_u = users[users["n_inter"] >= min_i]
        if c2.button("🎲 Random reader", use_container_width=True):
            st.session_state["uid"] = int(pool_u.sample(1).user_id.iloc[0])
        uid = c1.number_input("Reader ID", value=int(st.session_state.get("uid", pool_u.user_id.iloc[0])), step=1)

        x = recsys.user_vector(art, uid)
        if x is None:
            st.warning("This reader ID is not in the model (needs at least 5 interactions). "
                       "Try 🎲 Random reader, or use the 'By a Book I Like' tab.")
        else:
            u = users[users.user_id == uid].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🆔 Reader", int(uid))
            m2.metric("🌍 Country", str(u["country"]).title())
            m3.metric("🎂 Age", int(u["age"]) if pd.notna(u["age"]) else "n/a")
            m4.metric("📚 Books read", int(u["n_inter"]))

            hist = items.iloc[x.indices].copy()
            hist["w"] = x.data
            hist = hist.sort_values("w", ascending=False).head(6)
            with st.expander("📖 Reading history (top 6)"):
                hc = st.columns(6)
                for i, (_, row) in enumerate(hist.iterrows()):
                    with hc[i]:
                        mini_card(row, f"hist_{uid}_{i}_{row['isbn']}", h=150, show_rating=False)

            recs = recsys.recommend(art, x, **settings)
            render_recs(recs, "Picked for reader", f"#{int(uid)}", "ru")


# ─────────────────────────────────────────────────────────────
# PAGE: BROWSE LIBRARY
# ─────────────────────────────────────────────────────────────
elif page == "📚  Browse Library":
    page_title("Browse Library", "Explore the full catalog filtered by author, publisher or decade.")

    sidebar_label("Filters")
    sel_auth = st.sidebar.multiselect("Author", TOP_AUTHORS, default=[])
    sel_pub = st.sidebar.multiselect("Publisher", TOP_PUBS, default=[])
    sel_dec = st.sidebar.multiselect("Decade", DECADES, default=[])
    sidebar_label("Options")
    min_readers = st.sidebar.slider("Minimum readers", 5, 100, 20, step=5)
    num_books = st.sidebar.slider("Books to show", 6, 48, 18, step=6)
    sort_by = st.sidebar.selectbox("Sort by", ["Rating ↓ (Highest)", "Most Read", "Newest", "Title A→Z", "Title Z→A"])

    f = items[items["n_inter"] >= min_readers]
    if sel_auth:
        f = f[f["author"].isin(sel_auth)]
    if sel_pub:
        f = f[f["publisher"].isin(sel_pub)]
    if sel_dec:
        f = f[f["decade"].isin(sel_dec)]

    sort_map = {
        "Rating ↓ (Highest)": ("bayes_rating", False),
        "Most Read": ("n_inter", False),
        "Newest": ("year", False),
        "Title A→Z": ("title", True),
        "Title Z→A": ("title", False),
    }
    col_s, asc = sort_map[sort_by]
    f = f.sort_values(col_s, ascending=asc).head(num_books)

    active = [("Author", sel_auth), ("Publisher", sel_pub), ("Decade", sel_dec)]
    tags = "".join(f'<span class="am-chip">{esc(n)}: {esc(", ".join(v[:2]))}{"…" if len(v) > 2 else ""}</span>'
                   for n, v in active if v)
    if tags:
        st.markdown(f'<div style="margin-bottom:1.2rem;"><span style="color:#888899;font-size:.85rem;margin-right:8px;">Browsing:</span>{tags}</div>',
                    unsafe_allow_html=True)

    if len(f) == 0:
        st.warning("No books match these filters. Try lowering the minimum readers.")
    else:
        st.caption(f"{len(f)} books shown")
        cols = st.columns(3, gap="medium")
        for i, (_, row) in enumerate(f.iterrows()):
            with cols[i % 3]:
                browse_card(row, f"br_{i}_{row['isbn']}")
