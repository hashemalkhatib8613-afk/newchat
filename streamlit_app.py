import base64
import importlib
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR   = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "zain-logo.png"
DB_PATH   = APP_DIR / "zain_customer_360_ai_demo.db"


def get_logo_data_uri():
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

LOGO_DATA_URI = get_logo_data_uri()


def load_streamlit_secret():
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if key:
        os.environ["OPENAI_API_KEY"] = str(key)

load_streamlit_secret()

import class3_sql_agent_backend as backend  # noqa: E402
backend                   = importlib.reload(backend)
ask_sql_agent_payload     = backend.ask_sql_agent_payload
build_chart_from_question = backend.build_chart_from_question
execute_sql_query         = backend.execute_sql_query
get_database_overview     = backend.get_database_overview

_icon = str(LOGO_PATH) if LOGO_PATH.exists() else "📊"
st.set_page_config(
    page_title="Zain Customer 360 Copilot",
    page_icon=_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

CHART_TYPES = {
    "Bar":            "bar",
    "Horizontal bar": "horizontal_bar",
    "Pie":            "pie",
    "Doughnut":       "doughnut",
    "Line":           "line",
    "Area":           "area",
}

SUGGESTED_QUESTIONS = [
    "Find the top 10 customers with the highest churn score and explain why they are at risk.",
    "Which customer segments bring the most revenue in the last 6 months?",
    "What are the most common complaint categories and which ones are still unresolved?",
    "Which cities have the highest number of affected customers from network events?",
    "Which marketing campaigns have the best conversion rate?",
    "Show me the full profile, plan, complaints, churn risk, and recommended action for customer 42.",
    "Which customers have overdue invoices and high churn risk?",
    "Summarize recent support interactions by channel, reason, sentiment, and priority.",
    "Which plans have the highest average monthly revenue?",
    "Which high-value customers have negative support sentiment?",
]

NAV_ITEMS = [
    ("Chat",               "AI Chat",           "💬", "Ask business questions"),
    ("Analytics",          "Dynamic Analytics", "📊", "Filter KPIs and charts"),
    ("Chart Builder",      "Chart Builder",     "📈", "Create custom visuals"),
    ("SQL Query Builder",  "SQL Workspace",     "🧮", "Run safe SELECT queries"),
    ("Suggested Questions","Prompt Library",    "✨", "Ready-made use cases"),
]


# ─────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────
def ensure_state():
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"
    if "page" not in st.session_state:
        st.session_state.page = "Chat"
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = [{
            "id":         "chat_1",
            "title":      "New Chat",
            "messages":   [default_assistant_message()],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }]
        st.session_state.current_chat_id = "chat_1"
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    if "last_chart" not in st.session_state:
        st.session_state.last_chart = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    if "chat_search" not in st.session_state:
        st.session_state.chat_search = ""


def default_assistant_message():
    return {
        "role":    "assistant",
        "content": (
            "Hello. I am your Customer 360 AI Copilot. "
            "Ask me about customers, churn, complaints, billing, campaigns, "
            "support interactions, network events, or revenue performance."
        ),
        "sql": "",
        "ts":  datetime.now().strftime("%H:%M"),
    }


def title_from_question(question):
    cleaned = " ".join(str(question).split())
    return cleaned[:42] + "..." if len(cleaned) > 42 else cleaned or "New Chat"


def current_chat():
    ensure_state()
    for chat in st.session_state.chat_sessions:
        if chat["id"] == st.session_state.current_chat_id:
            return chat
    st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    return st.session_state.chat_sessions[0]


def create_new_chat():
    ensure_state()
    next_id = f"chat_{len(st.session_state.chat_sessions)+1}_{int(time.time())}"
    chat = {
        "id":         next_id,
        "title":      "New Chat",
        "messages":   [default_assistant_message()],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.chat_sessions.insert(0, chat)
    st.session_state.current_chat_id = next_id
    st.session_state.page = "Chat"


def delete_current_chat():
    ensure_state()
    if len(st.session_state.chat_sessions) == 1:
        st.session_state.chat_sessions[0]["title"]    = "New Chat"
        st.session_state.chat_sessions[0]["messages"] = [default_assistant_message()]
        return
    st.session_state.chat_sessions = [
        c for c in st.session_state.chat_sessions
        if c["id"] != st.session_state.current_chat_id
    ]
    st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]


# ─────────────────────────────────────────────────────────────
#  DATABASE HELPERS
# ─────────────────────────────────────────────────────────────
def db_connect():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(show_spinner=False)
def query_df(sql, params=()):
    with db_connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data(show_spinner=False)
def list_tables():
    return query_df(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )["name"].tolist()


@st.cache_data(show_spinner=False)
def table_columns(table_name):
    with db_connect() as conn:
        return pd.read_sql_query(f'PRAGMA table_info("{table_name}")', conn)


@st.cache_data(show_spinner=False)
def filter_options():
    return {
        "months": query_df(
            "SELECT DISTINCT summary_month FROM customer_monthly_summary ORDER BY summary_month"
        )["summary_month"].tolist(),
        "cities": query_df(
            "SELECT DISTINCT city FROM customers ORDER BY city"
        )["city"].dropna().tolist(),
        "segments": query_df(
            "SELECT DISTINCT customer_segment FROM customers ORDER BY customer_segment"
        )["customer_segment"].dropna().tolist(),
        "risk_levels": query_df(
            "SELECT DISTINCT risk_level FROM customer_churn_scores ORDER BY risk_level"
        )["risk_level"].dropna().tolist(),
        "service_types": query_df(
            "SELECT DISTINCT service_type FROM subscriptions ORDER BY service_type"
        )["service_type"].dropna().tolist(),
    }


# ─────────────────────────────────────────────────────────────
#  CSS — matches screenshot exactly
# ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
/* ── GLOBAL ── */
html, body, .stApp {
  background: #0e1117 !important;
  color: #f0f2f8 !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.block-container {
  padding: 1.5rem 2rem 5rem !important;
  max-width: 1400px !important;
}
h1,h2,h3,h4,h5,h6 { color: #f0f2f8 !important; }
p, label, span { color: #f0f2f8; }
a { color: #ff3347; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: #0a0b10 !important;
  border-right: 1px solid rgba(255,255,255,.07) !important;
}
section[data-testid="stSidebar"] > div {
  padding: 1rem .85rem 2rem !important;
}
section[data-testid="stSidebar"] * { color: #e8eaf2 !important; }

/* sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  min-height: 44px;
  border-radius: 10px !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  background: rgba(255,255,255,.04) !important;
  color: #c0c4d8 !important;
  font-size: .88rem;
  font-weight: 500;
  justify-content: flex-start;
  text-align: left;
  padding: .6rem .9rem;
  transition: all .15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,255,255,.08) !important;
  border-color: rgba(255,255,255,.14) !important;
  color: #fff !important;
}

/* ── BRAND CARD (top of sidebar) ── */
.brand-card {
  background: linear-gradient(135deg, rgba(180,20,28,.35), rgba(255,255,255,.04));
  border: 1px solid rgba(215,25,32,.30);
  border-radius: 14px;
  padding: 14px 12px 12px;
  margin-bottom: 16px;
}
.brand-title {
  font-size: .95rem;
  font-weight: 800;
  color: #fff !important;
  margin-bottom: 5px;
}
.brand-copy {
  font-size: .75rem;
  color: rgba(220,224,240,.60) !important;
  line-height: 1.5;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.chip {
  font-size: .65rem;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.14);
  color: #e0e4f0 !important;
  background: rgba(255,255,255,.07);
}

/* ── SIDEBAR SECTION LABEL ── */
.side-label {
  font-size: .65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .12em;
  color: rgba(200,204,220,.35) !important;
  margin: 14px 2px 6px;
}

/* ── ACTIVE NAV ITEM ── */
.active-nav {
  display: flex;
  align-items: center;
  gap: 8px;
  background: linear-gradient(135deg, rgba(215,25,32,.22), rgba(255,255,255,.05));
  border: 1px solid rgba(215,25,32,.40);
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 6px;
  color: #fff !important;
  font-size: .88rem;
  font-weight: 600;
  position: relative;
}
.active-nav::before {
  content: "";
  position: absolute;
  left: 0; top: 20%; bottom: 20%;
  width: 3px;
  border-radius: 0 2px 2px 0;
  background: #d71920;
}

/* ── HERO CARD ── */
.hero-card {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(180,20,28,.25), rgba(30,33,44,.95));
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px;
  padding: 28px 200px 28px 28px;
  margin-bottom: 20px;
}
.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(215,25,32,.18);
  border: 1px solid rgba(215,25,32,.35);
  border-radius: 999px;
  padding: 4px 12px;
  font-size: .70rem;
  font-weight: 800;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: #ff4455 !important;
  margin-bottom: 14px;
}
.hero-eyebrow::before {
  content: "";
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #ff3347;
}
.hero-title {
  font-size: clamp(1.7rem, 2.8vw, 2.8rem);
  font-weight: 900;
  letter-spacing: -.055em;
  line-height: 1.05;
  color: #fff !important;
  margin-bottom: 10px;
  max-width: 700px;
}
.hero-copy {
  font-size: .94rem;
  color: rgba(200,208,228,.70) !important;
  line-height: 1.6;
  max-width: 640px;
}
.hero-logo-box {
  position: absolute;
  right: 28px;
  top: 50%;
  transform: translateY(-50%);
  width: 120px;
  height: 120px;
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.hero-logo-img {
  position: absolute;
  right: 28px;
  top: 50%;
  transform: translateY(-50%);
  width: 120px;
  height: 120px;
  object-fit: contain;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.05);
  padding: 12px;
}
.hero-logo-text {
  font-size: .85rem;
  font-weight: 800;
  color: rgba(200,210,230,.25) !important;
  letter-spacing: .06em;
}
@media(max-width:700px) {
  .hero-card { padding: 20px !important; }
  .hero-logo-box, .hero-logo-img { display: none !important; }
}

/* ── PROMPT CARDS ── */
.prompt-card {
  position: relative;
  background: #181b24;
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 14px;
  padding: 14px 14px 20px;
  min-height: 110px;
  cursor: pointer;
  transition: border-color .18s, background .18s, transform .18s;
}
.prompt-card:hover {
  border-color: rgba(215,25,32,.40);
  background: #1e212e;
  transform: translateY(-2px);
}
.prompt-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  background: #d71920;
  border-radius: 6px;
  font-size: .68rem;
  font-weight: 800;
  color: #fff !important;
  margin-bottom: 10px;
}
.prompt-text {
  font-size: .875rem;
  color: rgba(210,216,235,.80) !important;
  line-height: 1.55;
}

/* ── KPI CARDS ── */
.kpi-card {
  background: #181b24;
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px;
  padding: 16px 14px 14px;
  min-height: 110px;
  transition: border-color .18s, transform .18s;
}
.kpi-card:hover {
  border-color: rgba(255,255,255,.14);
  transform: translateY(-2px);
}
.kpi-label {
  font-size: .70rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: rgba(160,168,196,.70) !important;
  margin-bottom: 6px;
}
.kpi-value {
  font-size: clamp(1.4rem, 2.4vw, 2rem);
  font-weight: 900;
  letter-spacing: -.045em;
  color: #f0f2f8 !important;
  line-height: 1;
  margin-bottom: 6px;
}
.kpi-badge {
  display: inline-flex;
  font-size: .65rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  margin-bottom: 4px;
}
.kpi-badge.good   { background: rgba(25,200,138,.12); color: #19c88a !important; border: 1px solid rgba(25,200,138,.22); }
.kpi-badge.warn   { background: rgba(245,183,49,.12); color: #f5b731 !important; border: 1px solid rgba(245,183,49,.22); }
.kpi-badge.danger { background: rgba(215,25,32,.12);  color: #ff3347 !important; border: 1px solid rgba(215,25,32,.22); }
.kpi-badge.info   { background: rgba(74,142,255,.12); color: #4a8eff !important; border: 1px solid rgba(74,142,255,.22); }
.kpi-badge.gray   { background: rgba(255,255,255,.07); color: #8892a4 !important; border: 1px solid rgba(255,255,255,.12); }
.kpi-note {
  font-size: .74rem;
  color: rgba(140,150,180,.65) !important;
  line-height: 1.4;
}

/* ── SECTION TITLE ── */
.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 12px 0 10px;
}
.section-title h3 {
  margin: 0;
  font-size: 1rem;
  font-weight: 800;
  color: rgba(200,208,230,.55) !important;
  text-transform: uppercase;
  letter-spacing: .06em;
}
.section-title span { font-size: .82rem; color: rgba(160,168,196,.55) !important; }

/* ── SOURCE BADGE ── */
.source-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: .65rem;
  font-weight: 800;
  letter-spacing: .07em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.source-sql   { background: rgba(74,142,255,.10);  color: #4a8eff !important; border: 1px solid rgba(74,142,255,.22); }
.source-rag   { background: rgba(25,200,138,.10);  color: #19c88a !important; border: 1px solid rgba(25,200,138,.22); }
.source-err   { background: rgba(215,25,32,.12);   color: #ff3347 !important; border: 1px solid rgba(215,25,32,.25); }
.source-cache { background: rgba(245,183,49,.10);  color: #f5b731 !important; border: 1px solid rgba(245,183,49,.22); }

/* ── MSG TIMESTAMP ── */
.msg-ts { font-size: .64rem; color: rgba(140,150,180,.50) !important; margin-top: 4px; }
.msg-ts.right { text-align: right; }
.msg-ts.bot   { text-align: left; }

/* ── CHAT MESSAGES ── */
.stChatMessage {
  background: #181b24 !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  border-radius: 16px !important;
  padding: 14px 16px !important;
}
[data-testid="stChatInput"] > div {
  background: #181b24 !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  border-radius: 14px !important;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: rgba(215,25,32,.50) !important;
  box-shadow: 0 0 0 3px rgba(215,25,32,.10) !important;
}

/* ── SHELL CARD ── */
.shell-card {
  background: #181b24;
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px;
  padding: 18px;
  margin-bottom: 16px;
}

/* ── BUTTONS ── */
.stButton > button,
.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
  min-height: 42px;
  border-radius: 10px !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  background: #1e212e !important;
  color: #e8eaf2 !important;
  font-weight: 600;
  transition: all .15s !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: rgba(215,25,32,.45) !important;
  color: #ff4455 !important;
}
.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, #d71920, #a01018) !important;
  border-color: rgba(255,255,255,.14) !important;
  color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #f02030, #c01520) !important;
}

/* ── INPUTS ── */
input, textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  border-radius: 10px !important;
  border-color: rgba(255,255,255,.12) !important;
  background: #181b24 !important;
  color: #f0f2f8 !important;
}
textarea { font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace !important; }

/* ── TABS ── */
div[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid rgba(255,255,255,.10) !important;
  gap: 4px;
}
div[data-testid="stTabs"] button {
  color: rgba(160,168,196,.60) !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 0 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: #f0f2f8 !important;
  border-bottom: 2px solid #d71920 !important;
}

/* ── METRICS ── */
div[data-testid="stMetric"] {
  background: #181b24;
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px;
  padding: 14px 16px;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
  font-size: .70rem !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: rgba(160,168,196,.70) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-size: 1.9rem !important;
  font-weight: 900 !important;
  letter-spacing: -.045em !important;
  color: #f0f2f8 !important;
}

/* ── DATAFRAME ── */
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,.08) !important;
}

/* ── EXPANDER ── */
div[data-testid="stExpander"] {
  background: #181b24 !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  border-radius: 12px !important;
  overflow: hidden;
}

/* ── ALERTS ── */
div[data-testid="stAlert"] {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,.08) !important;
}

/* ── MULTISELECT ── */
[data-baseweb="tag"] {
  border-radius: 999px !important;
  background: rgba(215,25,32,.12) !important;
  border: 1px solid rgba(215,25,32,.25) !important;
}
[data-baseweb="tag"] span { color: #ff4455 !important; font-weight: 700 !important; }

/* ── SQL EDITOR ── */
.sql-editor-wrap textarea {
  background: #06080e !important;
  color: #a8d8a0 !important;
  font-size: .82rem !important;
  line-height: 1.72 !important;
  border-color: rgba(168,216,160,.18) !important;
}

/* ── EMPTY STATE ── */
.empty-state {
  border: 1.5px dashed rgba(255,255,255,.10);
  border-radius: 16px;
  padding: 48px 24px;
  text-align: center;
  margin-top: 12px;
}
.empty-state-icon { font-size: 2.4rem; margin-bottom: 10px; opacity: .35; }
.empty-state-title { font-size: .95rem; font-weight: 800; color: rgba(160,168,196,.45) !important; margin-bottom: 4px; }
.empty-state-sub { font-size: .82rem; color: rgba(120,130,160,.40) !important; }

/* ── TIP CARD ── */
.tip-card {
  border-left: 3px solid #d71920;
  border-radius: 0 10px 10px 0;
  background: rgba(215,25,32,.07);
  padding: 10px 14px;
  font-size: .82rem;
  color: rgba(200,208,228,.70) !important;
  line-height: 1.5;
  margin-top: 8px;
}

/* ── SCROLL HINT ── */
.scroll-hint {
  text-align: center;
  padding: 6px;
  font-size: .75rem;
  color: rgba(140,150,180,.50) !important;
  animation: fadeup 1.5s ease-in-out infinite;
}
@keyframes fadeup {
  0%,100% { opacity:.3; transform:translateY(0); }
  50%      { opacity:.8; transform:translateY(-3px); }
}

/* ── MISC ── */
footer, #MainMenu { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border-radius: 999px; }
</style>
""", unsafe_allow_html=True)
    st.session_state.plot_template = "plotly_dark"


# ─────────────────────────────────────────────────────────────
#  UI COMPONENTS
# ─────────────────────────────────────────────────────────────
def hero(title, copy, eyebrow="Zain 360 Copilot"):
    if LOGO_DATA_URI:
        logo_html = f'<img class="hero-logo-img" src="{LOGO_DATA_URI}" alt="Zain">'
    else:
        logo_html = '<div class="hero-logo-box"><div class="hero-logo-text">⊙ZAIN</div></div>'

    st.markdown(f"""
<div class="hero-card">
  <div class="hero-eyebrow">{eyebrow}</div>
  <div class="hero-title">{title}</div>
  <div class="hero-copy">{copy}</div>
  {logo_html}
</div>""", unsafe_allow_html=True)


def kpi_card(label, value, note="", badge_text="", badge_type="gray"):
    badge_html = (
        f'<div class="kpi-badge {badge_type}">{badge_text}</div>'
        if badge_text else ""
    )
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {badge_html}
  <div class="kpi-note">{note}</div>
</div>""", unsafe_allow_html=True)


def source_badge(source_text):
    s = (source_text or "").lower()
    if "sql" in s:
        cls, label = "source-sql", "SQL Agent"
    elif "rag" in s or "cache" in s or "memory" in s:
        cls, label = "source-rag", "RAG Memory"
    elif "error" in s:
        cls, label = "source-err", "Error"
    else:
        cls, label = "source-cache", source_text or "Agent"
    st.markdown(
        f'<span class="source-badge {cls}">{label}</span>',
        unsafe_allow_html=True,
    )


def msg_timestamp(ts_str, align="right"):
    st.markdown(
        f'<div class="msg-ts {align}">{ts_str}</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon, title, subtitle):
    st.markdown(f"""
<div class="empty-state">
  <div class="empty-state-icon">{icon}</div>
  <div class="empty-state-title">{title}</div>
  <div class="empty-state-sub">{subtitle}</div>
</div>""", unsafe_allow_html=True)


def tip_card(text):
    st.markdown(f'<div class="tip-card">💡 {text}</div>', unsafe_allow_html=True)


def shell_start():
    st.markdown('<div class="shell-card">', unsafe_allow_html=True)


def shell_end():
    st.markdown("</div>", unsafe_allow_html=True)


def format_num(value, suffix=""):
    try:
        if pd.isna(value):
            return "0"
        value = float(value)
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M{suffix}"
        if abs(value) >= 1_000:
            return f"{value/1_000:.1f}K{suffix}"
        if float(value).is_integer():
            return f"{int(value):,}{suffix}"
        return f"{value:,.2f}{suffix}"
    except Exception:
        return str(value)


def plotly_layout(fig, height=400, legend=True):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1) if legend else None,
    )
    return fig


def build_chart(df, chart_type, title, x="label", y="value", color=None, height=410):
    if df is None or df.empty:
        empty_state("📭", "No data", "No data available for this visual.")
        return
    common = dict(template="plotly_dark", height=height, title=title)
    if chart_type == "pie":
        fig = px.pie(df, names=x, values=y, **common)
    elif chart_type == "doughnut":
        fig = px.pie(df, names=x, values=y, hole=0.55, **common)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, markers=True, color=color, **common)
    elif chart_type == "area":
        fig = px.area(df, x=x, y=y, color=color, **common)
    elif chart_type == "horizontal_bar":
        fig = px.bar(df, x=y, y=x, orientation="h", color=color, **common)
    else:
        fig = px.bar(df, x=x, y=y, color=color, **common)
    st.plotly_chart(plotly_layout(fig, height=height), use_container_width=True)


def render_chart(chart):
    rows = chart.get("rows") or []
    if not rows:
        st.warning(chart.get("summary") or "No matching data found.")
        return
    df = pd.DataFrame(rows)
    st.markdown(f"""
<div class="section-title">
  <h3>{chart.get('title','Chart')}</h3>
  <span>{chart.get('metric','Value')}</span>
</div>""", unsafe_allow_html=True)
    build_chart(df=df, chart_type=chart.get("chart_type", "bar"),
                title=chart.get("title", "Chart"), x="label", y="value")
    if chart.get("summary"):
        st.caption(chart["summary"])
    with st.expander("View chart data"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Export chart data",
            df.to_csv(index=False).encode("utf-8"),
            file_name="chart_data.csv", mime="text/csv", use_container_width=True,
        )


def chat_to_markdown(chat):
    lines = [f"# {chat['title']}", f"Created: {chat.get('created_at','')}", ""]
    for m in chat["messages"]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"## {role}")
        if m.get("source"):
            lines.append(f"Agent: {m['source']}")
        lines.append("")
        lines.append(m.get("content", ""))
        if m.get("sql"):
            lines += ["", "```sql", m["sql"], "```", ""]
    return "\n".join(lines)


def build_chat_history_context(chat, limit=8):
    history = []
    for m in chat.get("messages", [])[-limit:]:
        item = {"role": m.get("role", ""), "content": str(m.get("content", ""))[:4000]}
        if m.get("source"):
            item["source"] = m.get("source", "")
        if m.get("sql"):
            item["sql"] = m.get("sql", "")
        history.append(item)
    return history


def call_chat_backend(prompt, chat_history):
    try:
        return ask_sql_agent_payload(prompt, chat_history=chat_history)
    except TypeError as exc:
        if "chat_history" not in str(exc):
            raise
        return ask_sql_agent_payload(prompt)


def ask_and_store(prompt):
    prompt = str(prompt).strip()
    if not prompt:
        return
    chat = current_chat()
    if chat["title"] == "New Chat":
        chat["title"] = title_from_question(prompt)
    chat_history = build_chat_history_context(chat)
    chat["messages"].append({
        "role": "user", "content": prompt, "sql": "",
        "ts": datetime.now().strftime("%H:%M"),
    })
    try:
        with st.spinner("Analysing the database and preparing the answer…"):
            payload = call_chat_backend(prompt, chat_history)
        answer      = payload.get("answer", "No answer was returned.")
        sql         = payload.get("sql", "")
        src         = payload.get("source", "SQL Agent")
        matched_q   = payload.get("matched_question", "")
        match_score = payload.get("match_score", "")
    except Exception as exc:
        answer = (
            "I could not complete this request. "
            f"Details: {type(exc).__name__}: {exc}. "
            "Please confirm the OPENAI_API_KEY is configured."
        )
        sql = ""; src = "Error"; matched_q = ""; match_score = ""
    chat["messages"].append({
        "role": "assistant", "content": answer,
        "sql": sql, "source": src,
        "matched_question": matched_q, "match_score": match_score,
        "ts": datetime.now().strftime("%H:%M"),
    })


# ─────────────────────────────────────────────────────────────
#  SQL RUNNER
# ─────────────────────────────────────────────────────────────
def run_sql_callback(key_prefix):
    sql = st.session_state.get(f"{key_prefix}_sql_editor", "").strip()
    try:
        st.session_state[f"{key_prefix}_sql_result"] = execute_sql_query(sql)
        st.session_state[f"{key_prefix}_sql_error"]  = ""
    except Exception as exc:
        st.session_state[f"{key_prefix}_sql_result"] = None
        st.session_state[f"{key_prefix}_sql_error"]  = f"{type(exc).__name__}: {exc}"


def render_sql_runner(default_sql="", key_prefix="sql_runner"):
    editor_key = f"{key_prefix}_sql_editor"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = default_sql

    st.markdown('<div class="sql-editor-wrap">', unsafe_allow_html=True)
    st.text_area("SQL", height=200, key=editor_key,
                 help="Only safe read-only SELECT queries are allowed.")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        st.button("▶ Run Query", type="primary", key=f"{key_prefix}_run",
                  on_click=run_sql_callback, args=(key_prefix,), use_container_width=True)
    with c2:
        if st.button("✕ Clear", key=f"{key_prefix}_clear", use_container_width=True):
            st.session_state[f"{key_prefix}_sql_result"] = None
            st.session_state[f"{key_prefix}_sql_error"]  = ""
            st.rerun()

    error  = st.session_state.get(f"{key_prefix}_sql_error", "")
    result = st.session_state.get(f"{key_prefix}_sql_result")

    if error:
        st.error(f"Query failed: {error}")
    elif result:
        rows = result.get("rows", [])
        st.success(f"✓ Returned {len(rows)} row(s).")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Export as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="sql_result.csv", mime="text/csv", use_container_width=True,
            )
        else:
            st.info("Query ran successfully but returned no rows.")
        with st.expander("Executed SQL"):
            st.code(result.get("sql", ""), language="sql")


# ─────────────────────────────────────────────────────────────
#  ANALYTICS BUILDER
# ─────────────────────────────────────────────────────────────
def build_filtered_analytics(month_start, month_end, cities, segments, risk_levels, service_types):
    where  = ["m.summary_month BETWEEN ? AND ?"]
    params = [month_start, month_end]
    if cities:
        where.append("c.city IN (" + ",".join(["?"] * len(cities)) + ")")
        params.extend(cities)
    if segments:
        where.append("c.customer_segment IN (" + ",".join(["?"] * len(segments)) + ")")
        params.extend(segments)
    if risk_levels:
        where.append("ch.risk_level IN (" + ",".join(["?"] * len(risk_levels)) + ")")
        params.extend(risk_levels)
    if service_types:
        where.append("s.service_type IN (" + ",".join(["?"] * len(service_types)) + ")")
        params.extend(service_types)
    ws = " AND ".join(where)

    kpi_sql = f"""
        SELECT
            COUNT(DISTINCT c.customer_id)      AS customers,
            ROUND(SUM(m.monthly_revenue),2)    AS revenue,
            ROUND(AVG(ch.churn_score),4)       AS avg_churn,
            ROUND(AVG(m.data_usage_gb),2)      AS avg_data,
            COUNT(DISTINCT comp.complaint_id)  AS complaints,
            COUNT(DISTINCT si.interaction_id)  AS support
        FROM customer_monthly_summary m
        JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = m.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        LEFT JOIN complaints comp ON comp.customer_id = c.customer_id
            AND strftime('%Y-%m', comp.created_at) BETWEEN ? AND ?
        LEFT JOIN support_interactions si ON si.customer_id = c.customer_id
            AND strftime('%Y-%m', si.interaction_date) BETWEEN ? AND ?
        WHERE {ws}"""
    extra  = [month_start, month_end, month_start, month_end]
    kpi_df = query_df(kpi_sql, tuple(extra + params))

    revenue_sql = f"""
        SELECT m.summary_month AS label, ROUND(SUM(m.monthly_revenue),2) AS value
        FROM customer_monthly_summary m
        JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = m.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {ws} GROUP BY m.summary_month ORDER BY m.summary_month"""
    risk_sql = f"""
        SELECT ch.risk_level AS label, COUNT(*) AS value
        FROM customer_churn_scores ch
        JOIN customers c ON c.customer_id = ch.customer_id
        LEFT JOIN customer_monthly_summary m ON m.customer_id = c.customer_id
            AND m.summary_month BETWEEN ? AND ?
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {ws} GROUP BY ch.risk_level"""
    seg_sql = f"""
        SELECT c.customer_segment AS label, ROUND(SUM(m.monthly_revenue),2) AS value
        FROM customer_monthly_summary m
        JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = m.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {ws} GROUP BY c.customer_segment ORDER BY value DESC"""
    city_sql = f"""
        SELECT c.city AS label, COUNT(DISTINCT c.customer_id) AS value
        FROM customers c
        JOIN customer_monthly_summary m ON m.customer_id = c.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {ws} GROUP BY c.city ORDER BY value DESC LIMIT 12"""
    usage_sql = f"""
        SELECT m.summary_month AS label, ROUND(AVG(m.data_usage_gb),2) AS value
        FROM customer_monthly_summary m
        JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = m.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {ws} GROUP BY m.summary_month ORDER BY m.summary_month"""
    cust_sql = f"""
        SELECT c.customer_segment AS label, COUNT(DISTINCT c.customer_id) AS value
        FROM customers c
        JOIN customer_monthly_summary m ON m.customer_id = c.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {ws} GROUP BY c.customer_segment"""

    p  = tuple(params)
    p2 = tuple([month_start, month_end] + params)
    return {
        "kpi":     kpi_df,
        "revenue": query_df(revenue_sql, p),
        "risk":    query_df(risk_sql,    p2),
        "seg":     query_df(seg_sql,     p),
        "city":    query_df(city_sql,    p),
        "usage":   query_df(usage_sql,   p),
        "cust":    query_df(cust_sql,    p),
    }


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    ensure_state()
    with st.sidebar:
        # Brand card
        st.markdown("""
<div class="brand-card">
  <div class="brand-title">Customer 360 AI Copilot</div>
  <div class="brand-copy">Premium analytics workspace for customers, churn, revenue,
    complaints, support, campaigns, and network signals.</div>
  <div class="chip-row">
    <span class="chip">SQL-backed</span>
    <span class="chip">AI Chat</span>
    <span class="chip">Dynamic BI</span>
  </div>
</div>""", unsafe_allow_html=True)

        # Active page indicator
        page = st.session_state.page
        for page_key, label, icon, sub in NAV_ITEMS:
            if page_key == page:
                st.markdown(
                    f'<div class="active-nav">{icon} {label}</div>',
                    unsafe_allow_html=True,
                )
                break

        # New chat button
        if st.button("＋  New Chat", key="new_chat_btn", use_container_width=True):
            create_new_chat()
            st.rerun()

        # Saved chats
        st.markdown('<div class="side-label">Saved Chats</div>', unsafe_allow_html=True)

        search_q = st.text_input(
            "Search chats", placeholder="🔍  Search…",
            label_visibility="collapsed", key="chat_search_input",
        )
        st.session_state.chat_search = search_q.lower().strip()

        for chat in st.session_state.chat_sessions:
            if (st.session_state.chat_search
                    and st.session_state.chat_search not in chat["title"].lower()):
                continue
            label      = chat["title"] if chat["title"] != "New Chat" else "💬 New Chat"
            is_current = chat["id"] == st.session_state.current_chat_id
            btn_label  = f"→ {label}" if is_current else f"   {label}"
            if st.button(btn_label, key=f"chat_sel_{chat['id']}", use_container_width=True):
                st.session_state.current_chat_id = chat["id"]
                st.session_state.page = "Chat"
                st.rerun()

        if st.button("🗑  Delete current chat", key="del_chat_btn", use_container_width=True):
            delete_current_chat()
            st.rerun()

        # Navigation
        st.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
        for page_key, label, icon, sub in NAV_ITEMS:
            if st.button(f"{icon}  {label}", key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

        # DB schema
        st.markdown('<div class="side-label">Database</div>', unsafe_allow_html=True)
        with st.expander("📂 Tables"):
            try:
                for t in list_tables():
                    ncols = len(table_columns(t))
                    st.markdown(f"**{t}** — {ncols} cols")
            except Exception:
                st.caption("Could not load schema.")


# ─────────────────────────────────────────────────────────────
#  PAGE: CHAT
# ─────────────────────────────────────────────────────────────
def page_chat():
    hero(
        "Customer 360 Chat",
        "Ask direct business questions. Your chat sessions are saved during this browser session.",
        eyebrow="💬 AI Chat",
    )

    st.markdown("""
<div class="section-title">
  <h3>Quick prompts</h3>
  <span>Start with a common telecom question</span>
</div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    for i, q in enumerate(SUGGESTED_QUESTIONS[:4]):
        with cols[i]:
            st.markdown(
                f'<div class="prompt-card">'
                f'<div class="prompt-num">{i+1}</div>'
                f'<div class="prompt-text">{q}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Ask →", key=f"qprompt_{i}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()

    st.divider()

    chat = current_chat()
    c_exp, c_title = st.columns([1, 6])
    with c_exp:
        st.download_button(
            "⬇ Export chat",
            data=chat_to_markdown(chat).encode("utf-8"),
            file_name=f"{chat['title'][:30]}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with c_title:
        st.markdown(
            f"<div style='padding:.55rem 0;font-size:.80rem;"
            f"color:rgba(140,150,180,.55)'>Session started {chat.get('created_at','')}</div>",
            unsafe_allow_html=True,
        )

    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("source"):
                source_badge(msg["source"])
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("🔍 View SQL"):
                    st.code(msg["sql"], language="sql")
            if msg.get("ts"):
                align = "right" if msg["role"] == "user" else "bot"
                msg_timestamp(msg["ts"], align)

    if len(chat["messages"]) > 4:
        st.markdown('<div class="scroll-hint">▼</div>', unsafe_allow_html=True)

    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
        ask_and_store(prompt)
        st.rerun()

    if prompt := st.chat_input(
        "Ask about churn, customers, revenue, billing, campaigns, complaints, or network impact…"
    ):
        ask_and_store(prompt)
        st.rerun()


# ─────────────────────────────────────────────────────────────
#  PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────
def page_analytics():
    hero(
        "Dynamic Analytics",
        "Adjust date ranges, customer segments, risk levels, cities, services, and chart styles. "
        "Export the filtered dataset when needed.",
        eyebrow="📊 Interactive BI",
    )

    opts   = filter_options()
    months = opts.get("months", [])
    if not months:
        st.warning("No monthly summary data found.")
        return

    shell_start()
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 2, 2, 2])
    with fc1:
        idx = st.select_slider("Month range", options=months, value=(months[0], months[-1]))
        month_start, month_end = idx
    with fc2:
        cities = st.multiselect("Cities", opts["cities"], placeholder="All cities")
    with fc3:
        segments = st.multiselect("Customer segments", opts["segments"], placeholder="All segments")
    with fc4:
        risk_levels = st.multiselect("Risk levels", opts["risk_levels"], placeholder="All levels")
    with fc5:
        service_types = st.multiselect("Service types", opts["service_types"], placeholder="All services")

    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        if st.button("⚡ High-risk only", use_container_width=True):
            risk_levels = ["High"]
    with qc2:
        if st.button("👑 VIP customers", use_container_width=True):
            segments = ["VIP"]
    with qc3:
        if st.button("🏙 Amman view", use_container_width=True):
            cities = ["Amman"]
    with qc4:
        if st.button("↺ Reset all", use_container_width=True):
            cities = []; segments = []; risk_levels = []; service_types = []
    shell_end()

    with st.spinner("Loading analytics…"):
        try:
            data = build_filtered_analytics(
                month_start, month_end, cities, segments, risk_levels, service_types
            )
        except Exception as e:
            st.error(f"Could not load analytics: {e}")
            return

    kpi = data["kpi"].iloc[0] if not data["kpi"].empty else {}
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        kpi_card("Customers",  format_num(kpi.get("customers", 0)),
                 "Distinct filtered customers", badge_text="Live", badge_type="good")
    with k2:
        kpi_card("Revenue",    format_num(kpi.get("revenue", 0)) + " JOD",
                 "Filtered monthly revenue", badge_text="↑ 4%", badge_type="good")
    with k3:
        kpi_card("Avg churn",  f"{float(kpi.get('avg_churn', 0)):.2f}",
                 "Average churn score", badge_text="High", badge_type="warn")
    with k4:
        kpi_card("Avg data",   format_num(kpi.get("avg_data", 0)) + " GB",
                 "Avg monthly usage", badge_text="Stable", badge_type="info")
    with k5:
        kpi_card("Complaints", format_num(kpi.get("complaints", 0)),
                 "Within selected months", badge_text="↑ 12%", badge_type="danger")
    with k6:
        kpi_card("Support",    format_num(kpi.get("support", 0)),
                 "Interaction count", badge_text="Normal", badge_type="gray")

    _, cstyle_col = st.columns([5, 1])
    with cstyle_col:
        chart_style = st.selectbox("Chart style", list(CHART_TYPES.keys()), label_visibility="collapsed")
    ct = CHART_TYPES.get(chart_style, "bar")

    tab_rev, tab_risk, tab_seg, tab_city, tab_usage, tab_cust = st.tabs(
        ["Revenue", "Risk", "Segments", "City", "Usage", "Customers"]
    )
    with tab_rev:
        st.markdown('<div class="section-title"><h3>Revenue trend by month</h3></div>',
                    unsafe_allow_html=True)
        build_chart(data["revenue"], ct, "Revenue trend by month")
        if not data["revenue"].empty:
            st.download_button("⬇ Export", data["revenue"].to_csv(index=False).encode(),
                               file_name="revenue.csv", mime="text/csv")
    with tab_risk:
        st.markdown('<div class="section-title"><h3>Churn risk distribution</h3></div>',
                    unsafe_allow_html=True)
        build_chart(data["risk"], "pie", "Risk level breakdown")
    with tab_seg:
        st.markdown('<div class="section-title"><h3>Revenue by customer segment</h3></div>',
                    unsafe_allow_html=True)
        build_chart(data["seg"], ct, "Revenue by segment")
    with tab_city:
        st.markdown('<div class="section-title"><h3>Customers by city (top 12)</h3></div>',
                    unsafe_allow_html=True)
        build_chart(data["city"], "horizontal_bar", "Customers by city")
    with tab_usage:
        st.markdown('<div class="section-title"><h3>Average data usage trend</h3></div>',
                    unsafe_allow_html=True)
        build_chart(data["usage"], "line", "Avg data usage by month")
    with tab_cust:
        st.markdown('<div class="section-title"><h3>Customer count by segment</h3></div>',
                    unsafe_allow_html=True)
        build_chart(data["cust"], "doughnut", "Customers by segment")


# ─────────────────────────────────────────────────────────────
#  PAGE: CHART BUILDER
# ─────────────────────────────────────────────────────────────
def page_chart_builder():
    hero(
        "Chart Builder",
        "Describe the chart you want in business language. "
        "The app plans a safe read-only query and turns the result into a visual.",
        eyebrow="📈 Natural-Language Visuals",
    )

    shell_start()
    question = st.text_area(
        "Chart inquiry",
        placeholder="e.g. Show monthly churn rate by city for the last 6 months",
        height=110, key="chart_question",
    )
    c1, c2 = st.columns([2, 5])
    with c1:
        chart_type_label = st.selectbox("Chart type", list(CHART_TYPES.keys()), key="chart_type_sel")
    with c2:
        tip_card("Ask for one clear metric — e.g. conversion by campaign, churn by city, or complaints by category.")
    submitted = st.button("🎨  Create Chart", type="primary")
    shell_end()

    if submitted and question.strip():
        with st.spinner("Building chart…"):
            try:
                chart = build_chart_from_question(
                    question.strip(),
                    chart_type=CHART_TYPES.get(chart_type_label, "bar"),
                )
                st.session_state.last_chart = chart
            except Exception as exc:
                st.error(f"Chart generation failed: {exc}")
                st.session_state.last_chart = None
    elif submitted:
        st.warning("Please enter a chart description.")

    if st.session_state.get("last_chart"):
        render_chart(st.session_state.last_chart)
    else:
        empty_state("📊", "No chart yet",
                    "Describe a chart above and click Create Chart to generate a visual.")


# ─────────────────────────────────────────────────────────────
#  PAGE: SQL WORKSPACE
# ─────────────────────────────────────────────────────────────
def page_sql():
    hero(
        "SQL Workspace",
        "Run safe read-only SELECT queries directly against the customer database. "
        "Results export as CSV.",
        eyebrow="🧮 SQL Workspace",
    )

    with st.expander("📂 Schema reference"):
        try:
            tables = list_tables()
            tcols  = st.columns(min(len(tables), 4))
            for i, t in enumerate(tables):
                with tcols[i % 4]:
                    cols_df = table_columns(t)
                    st.markdown(f"**{t}**")
                    for _, row in cols_df.iterrows():
                        st.markdown(
                            f"<span style='font-size:.76rem;color:rgba(140,150,180,.65)'>"
                            f"{row['name']} ({row['type']})</span>",
                            unsafe_allow_html=True,
                        )
        except Exception:
            st.caption("Schema unavailable.")

    shell_start()
    render_sql_runner(
        default_sql=(
            "SELECT c.customer_id, c.name, c.city,\n"
            "       cs.churn_score, cs.risk_level\n"
            "FROM customers c\n"
            "JOIN customer_churn_scores cs ON c.customer_id = cs.customer_id\n"
            "ORDER BY cs.churn_score DESC\n"
            "LIMIT 10;"
        )
    )
    shell_end()


# ─────────────────────────────────────────────────────────────
#  PAGE: PROMPT LIBRARY
# ─────────────────────────────────────────────────────────────
def page_suggested():
    hero(
        "Ready-Made Use Cases",
        "Click any prompt to send it directly to the AI Chat. "
        "Covers churn, revenue, complaints, campaigns, and network signals.",
        eyebrow="✨ Prompt Library",
    )

    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            st.markdown(
                f'<div class="prompt-card">'
                f'<div class="prompt-num">{i+1}</div>'
                f'<div class="prompt-text">{q}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Send to chat →", key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.session_state.page = "Chat"
                st.rerun()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    ensure_state()
    inject_css()
    render_sidebar()

    page = st.session_state.page
    if page == "Chat":
        page_chat()
    elif page == "Analytics":
        page_analytics()
    elif page == "Chart Builder":
        page_chart_builder()
    elif page == "SQL Query Builder":
        page_sql()
    elif page == "Suggested Questions":
        page_suggested()
    else:
        page_chat()


if __name__ == "__main__":
    main()
