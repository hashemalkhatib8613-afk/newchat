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

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "zain-logo.png"
DB_PATH = APP_DIR / "zain_customer_360_ai_demo.db"


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

backend = importlib.reload(backend)
ask_sql_agent_payload = backend.ask_sql_agent_payload
build_chart_from_question = backend.build_chart_from_question
execute_sql_query = backend.execute_sql_query
get_database_overview = backend.get_database_overview


st.set_page_config(
    page_title="Zain Customer 360 Copilot",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHART_TYPES = {
    "Bar": "bar",
    "Horizontal bar": "horizontal_bar",
    "Pie": "pie",
    "Doughnut": "doughnut",
    "Line": "line",
    "Area": "area",
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
    ("Customer Insights",  "Customer 360",      "👤", "Single customer view"),
    ("Data Catalog",       "Data Catalog",      "🗄", "Schema explorer"),
]


# ─────────────────────────── STATE ────────────────────────────

def ensure_state():
    if "page" not in st.session_state:
        st.session_state.page = "Chat"
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = [
            {
                "id": "chat_1",
                "title": "New Chat",
                "messages": [default_assistant_message()],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        ]
        st.session_state.current_chat_id = "chat_1"
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    if "last_chart" not in st.session_state:
        st.session_state.last_chart = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    st.session_state.plot_template = "plotly_dark"


def default_assistant_message():
    return {
        "role": "assistant",
        "content": (
            "Hello. I am your Customer 360 AI Copilot. Ask me about customers, churn, complaints, billing, "
            "campaigns, support interactions, network events, or revenue performance."
        ),
        "sql": "",
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
    next_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{int(time.time())}"
    chat = {
        "id": next_id,
        "title": "New Chat",
        "messages": [default_assistant_message()],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.chat_sessions.insert(0, chat)
    st.session_state.current_chat_id = next_id
    st.session_state.page = "Chat"


def delete_current_chat():
    ensure_state()
    if len(st.session_state.chat_sessions) == 1:
        st.session_state.chat_sessions[0]["title"] = "New Chat"
        st.session_state.chat_sessions[0]["messages"] = [default_assistant_message()]
        return
    st.session_state.chat_sessions = [
        c for c in st.session_state.chat_sessions if c["id"] != st.session_state.current_chat_id
    ]
    st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]


# ─────────────────────────── DB ────────────────────────────

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
    return query_df("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")["name"].tolist()


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
        "cities": query_df("SELECT DISTINCT city FROM customers ORDER BY city")["city"].dropna().tolist(),
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


# ─────────────────────────── CSS ────────────────────────────

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
          --bg:        #07080E;
          --bg2:       #0C0E17;
          --s1:        #111421;
          --s2:        #171A27;
          --s3:        #1C2030;
          --brand:       #63CEF8;
          --brand-a:     rgba(99,206,248,.14);
          --brand-b:     rgba(99,206,248,.30);
          --brand-c:     rgba(145,60,175,.08);
          --txt:       #E8EBF6;
          --m1:        #8591AB;
          --m2:        #545E78;
          --m3:        #353D57;
          --ok:        #17B978;
          --warn:      #E8960F;
          --bad:       #E84040;
          --ok-a:      rgba(23,185,120,.13);
          --warn-a:    rgba(232,150,15,.13);
          --bad-a:     rgba(232,64,64,.13);
          --f:         'Plus Jakarta Sans', sans-serif;
          --mono:      'JetBrains Mono', monospace;
          --r:         10px;
          --rl:        14px;
          --rxl:       18px;
        }

        *, *::before, *::after { box-sizing: border-box; }

        html, body, .stApp {
          font-family: var(--f) !important;
          background: var(--bg) !important;
          color: var(--txt) !important;
        }

        /* Subtle logo-color ambient on the page bg */
        .stApp {
          background:
            radial-gradient(ellipse 700px 500px at -5% -10%, rgba(99,206,248,.16) 0%, transparent 55%),
            radial-gradient(ellipse 500px 400px at 105% 5%,  rgba(145,60,175,.12) 0%, transparent 50%),
            var(--bg) !important;
        }

        /* ── Block container ── */
        .block-container {
          padding: 1.25rem 2rem 5rem !important;
          max-width: 1560px !important;
        }
        @media (max-width: 900px) {
          .block-container { padding: 1rem 1rem 5rem !important; }
        }

        /* ── Global type ── */
        h1, h2, h3, h4, h5, h6 {
          font-family: var(--f) !important;
          color: var(--txt) !important;
          letter-spacing: -.03em;
        }
        p, label, span, li { color: var(--txt); }
        a { color: var(--brand); text-decoration: none; }

        /* ─────────── SIDEBAR ─────────── */
        section[data-testid="stSidebar"] {
          background:
            radial-gradient(ellipse 280px 220px at 50% -8%, rgba(99,206,248,.20) 0%, transparent 55%),
            linear-gradient(180deg, #090B14 0%, #06070D 100%) !important;
          border-right: 1px solid rgba(255,255,255,.06) !important;
        }
        section[data-testid="stSidebar"] > div {
          padding: 1rem .9rem 1.5rem !important;
        }
        section[data-testid="stSidebar"] * { color: var(--txt) !important; }

        /* Sidebar buttons — shared base */
        [data-testid="stSidebar"] .stButton > button {
          width: 100%;
          min-height: 38px;
          border-radius: var(--rl) !important;
          border: 1px solid rgba(255,255,255,.07) !important;
          background: rgba(255,255,255,.04) !important;
          color: var(--m1) !important;
          font-family: var(--f) !important;
          font-weight: 500 !important;
          font-size: .82rem !important;
          justify-content: flex-start !important;
          text-align: left !important;
          padding: .5rem .8rem !important;
          transition: background .14s, border-color .14s, color .14s !important;
          box-shadow: none !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
          border-color: rgba(255,255,255,.14) !important;
          background: rgba(255,255,255,.08) !important;
          color: var(--txt) !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
          background: linear-gradient(135deg, var(--brand), #8C3AAF) !important;
          border-color: rgba(255,255,255,.12) !important;
          color: #fff !important;
          font-weight: 600 !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
          background: linear-gradient(135deg, #8BDFFF, #6E66D8) !important;
        }

        /* ── Sidebar brand block ── */
        .sb-brand {
          display: flex;
          align-items: center;
          gap: .7rem;
          padding: .8rem .9rem;
          background: var(--brand-a);
          border: 1px solid var(--brand-b);
          border-radius: var(--rxl);
          margin-bottom: 1rem;
        }
        .sb-brand-mark {
          width: 34px; height: 34px;
          border-radius: 9px;
          background: linear-gradient(135deg, var(--brand), #6E66D8);
          display: flex; align-items: center; justify-content: center;
          font-size: 1rem; flex-shrink: 0;
          color: #fff !important;
          font-weight: 800;
        }
        .sb-brand-name {
          font-size: .9rem; font-weight: 700;
          color: #fff !important; line-height: 1.25;
        }
        .sb-brand-sub {
          font-size: .69rem; color: rgba(255,255,255,.45) !important;
        }

        /* Sidebar section labels */
        .sb-lbl {
          font-size: .64rem; font-weight: 700;
          letter-spacing: .14em; text-transform: uppercase;
          color: var(--m3) !important;
          margin: .9rem .1rem .35rem;
          display: block;
        }

        /* Active page pill */
        .sb-active {
          display: flex; align-items: center; gap: .6rem;
          padding: .6rem .85rem;
          background: var(--brand-a);
          border: 1px solid var(--brand-b);
          border-radius: var(--rl);
          margin-bottom: .7rem;
        }
        .sb-active-name {
          font-size: .84rem; font-weight: 700; color: #fff !important;
        }
        .sb-active-desc {
          font-size: .68rem; color: rgba(255,255,255,.45) !important;
        }
        .sb-dot {
          width: 6px; height: 6px; border-radius: 50%;
          background: var(--brand);
          box-shadow: 0 0 5px var(--brand);
          flex-shrink: 0; margin-left: auto;
        }

        /* ─────────── HERO BANNER ─────────── */
        .hero {
          position: relative; overflow: hidden;
          border: 1px solid rgba(255,255,255,.09);
          border-radius: var(--rxl);
          padding: 1.35rem 1.6rem;
          background:
            radial-gradient(ellipse 55% 130% at 0% 50%, rgba(99,206,248,.18) 0%, transparent 55%),
            linear-gradient(135deg, var(--s1), var(--s2));
          margin-bottom: 1.25rem;
        }
        /* Fine crosshatch texture */
        .hero::before {
          content: "";
          position: absolute; inset: 0;
          background-image: repeating-linear-gradient(
            45deg,
            rgba(255,255,255,.012) 0, rgba(255,255,255,.012) 1px,
            transparent 0, transparent 50%
          );
          background-size: 12px 12px;
          pointer-events: none;
        }
        .hero-eyebrow {
          display: inline-flex; align-items: center; gap: .4rem;
          padding: .26rem .6rem;
          background: var(--brand-a);
          border: 1px solid var(--brand-b);
          border-radius: 999px;
          color: #8BDFFF !important;
          font-size: .66rem; font-weight: 700;
          letter-spacing: .11em; text-transform: uppercase;
          margin-bottom: .65rem;
        }
        .hero-title {
          font-family: var(--f);
          font-size: clamp(1.55rem, 2.6vw, 2.4rem);
          font-weight: 800; letter-spacing: -.045em; line-height: 1.05;
          color: #fff !important; margin: 0 0 .45rem;
        }
        .hero-copy {
          font-size: .91rem; color: var(--m1) !important;
          line-height: 1.6; max-width: 720px; margin: 0;
        }
        .hero-logo-wrap {
          position: absolute; right: 1.6rem; top: 50%;
          transform: translateY(-50%);
          width: clamp(54px, 6.5vw, 82px);
          height: clamp(54px, 6.5vw, 82px);
          background: rgba(255,255,255,.04);
          border: 1px solid rgba(255,255,255,.10);
          border-radius: var(--rl);
          display: flex; align-items: center; justify-content: center;
        }
        .hero-logo-wrap img { width: 65%; opacity: .68; }
        @media (max-width: 640px) { .hero-logo-wrap { display: none; } }

        /* ─────────── KPI CARDS ─────────── */
        .kpi-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: .7rem; margin-bottom: 1.15rem;
        }
        .kpi {
          border: 1px solid rgba(255,255,255,.07);
          border-radius: var(--rxl);
          padding: .95rem 1.05rem;
          background: var(--s1);
          position: relative; overflow: hidden;
          transition: border-color .18s;
        }
        .kpi::before {
          content: "";
          position: absolute; top: 0; left: 0; right: 0; height: 2px;
          opacity: 0; transition: opacity .18s;
        }
        .kpi.ok::before  { background: var(--ok);   opacity: 1; }
        .kpi.warn::before{ background: var(--warn);  opacity: 1; }
        .kpi.bad::before { background: var(--bad);   opacity: 1; }
        .kpi.neu::before { background: rgba(255,255,255,.18); opacity: 1; }
        .kpi:hover { border-color: rgba(255,255,255,.14); }
        .kpi-label {
          font-size: .66rem; font-weight: 700;
          letter-spacing: .1em; text-transform: uppercase;
          color: var(--m2) !important; margin-bottom: .5rem;
        }
        .kpi-value {
          font-family: var(--mono);
          font-size: clamp(1.25rem, 2.2vw, 1.8rem);
          font-weight: 600; letter-spacing: -.03em;
          color: #fff !important; line-height: 1; margin-bottom: .28rem;
        }
        .kpi-value.ok   { color: var(--ok)   !important; }
        .kpi-value.warn { color: var(--warn)  !important; }
        .kpi-value.bad  { color: var(--bad)   !important; }
        .kpi-note { font-size: .74rem; color: var(--m2) !important; line-height: 1.35; }

        /* ─────────── SECTION HEADER ─────────── */
        .section-hd {
          display: flex; align-items: baseline;
          justify-content: space-between; gap: 1rem;
          margin: .2rem 0 .85rem;
        }
        .section-hd h3 {
          font-size: .98rem; font-weight: 700; margin: 0;
          color: var(--txt) !important;
        }
        .section-hd span { font-size: .77rem; color: var(--m2) !important; }

        /* ─────────── CARDS ─────────── */
        .card {
          background: var(--s1);
          border: 1px solid rgba(255,255,255,.07);
          border-radius: var(--rxl);
          padding: 1.1rem; margin-bottom: .8rem;
        }

        /* ─────────── PROMPT CARDS ─────────── */
        .prompt-card {
          border: 1px solid rgba(255,255,255,.07);
          border-radius: var(--rxl);
          padding: .9rem 1rem;
          background: var(--s1);
          min-height: 115px;
          transition: border-color .16s, background .16s;
        }
        .prompt-card:hover {
          border-color: var(--brand-b);
          background: var(--s2);
        }
        .prompt-num {
          font-size: .63rem; font-weight: 700;
          letter-spacing: .12em; text-transform: uppercase;
          color: var(--brand) !important; margin-bottom: .4rem;
        }
        .prompt-card p {
          margin: 0; color: var(--m1) !important;
          font-size: .83rem; line-height: 1.5;
        }

        /* ─────────── BADGES / PILLS ─────────── */
        .badge {
          display: inline-flex; align-items: center;
          padding: .22rem .52rem; border-radius: 999px;
          background: var(--brand-a); border: 1px solid var(--brand-b);
          color: #FF6B80 !important;
          font-size: .68rem; font-weight: 600;
          margin: .1rem .15rem .1rem 0;
        }
        .badge-ok   { background: var(--ok-a);   border-color: rgba(23,185,120,.25); color: var(--ok)   !important; }
        .badge-warn { background: var(--warn-a);  border-color: rgba(232,150,15,.25); color: var(--warn)  !important; }
        .badge-bad  { background: var(--bad-a);   border-color: rgba(232,64,64,.25);  color: var(--bad)   !important; }

        /* ─────────── BUTTONS (main content) ─────────── */
        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] > button {
          font-family: var(--f) !important;
          font-weight: 500 !important; min-height: 40px;
          border-radius: var(--rl) !important;
          border: 1px solid rgba(255,255,255,.10) !important;
          background: var(--s2) !important;
          color: var(--txt) !important;
          box-shadow: none !important;
          transition: border-color .14s, background .14s, transform .1s !important;
          font-size: .85rem !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
          border-color: var(--brand-b) !important;
          background: var(--s1) !important;
          transform: translateY(-1px) !important;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button {
          background: linear-gradient(135deg, var(--brand), #8C3AAF) !important;
          border-color: rgba(255,255,255,.12) !important;
          color: #fff !important; font-weight: 600 !important;
        }
        .stButton > button[kind="primary"]:hover {
          background: linear-gradient(135deg, #8BDFFF, #6E66D8) !important;
        }

        /* ─────────── INPUTS ─────────── */
        input, textarea,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
          font-family: var(--f) !important;
          background: var(--s2) !important;
          border: 1px solid rgba(255,255,255,.10) !important;
          border-radius: var(--rl) !important;
          color: var(--txt) !important;
        }
        textarea {
          font-family: var(--mono) !important;
          font-size: .83rem !important;
        }

        /* ─────────── TABS ─────────── */
        div[data-testid="stTabs"] [role="tablist"] {
          gap: .25rem;
          border-bottom: 1px solid rgba(255,255,255,.07) !important;
        }
        div[data-testid="stTabs"] button[role="tab"] {
          font-family: var(--f) !important;
          font-size: .79rem !important; font-weight: 600 !important;
          color: var(--m2) !important;
          border-radius: 8px 8px 0 0 !important;
          padding: .42rem .88rem !important;
          border: 1px solid transparent !important;
          border-bottom: none !important;
          background: transparent !important;
          transition: color .13s, background .13s !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover {
          color: var(--txt) !important; background: var(--s1) !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
          color: #fff !important;
          background: var(--s1) !important;
          border-color: rgba(255,255,255,.07) !important;
        }

        /* ─────────── CHAT ─────────── */
        .stChatMessage {
          background: var(--s1) !important;
          border: 1px solid rgba(255,255,255,.07) !important;
          border-radius: var(--rxl) !important;
          padding: .8rem !important;
        }
        [data-testid="stChatMessageContent"] p {
          font-size: .91rem; line-height: 1.65;
        }
        [data-testid="stChatInput"] {
          border-radius: var(--rxl) !important;
          background: var(--s2) !important;
          border: 1px solid rgba(255,255,255,.10) !important;
        }
        [data-testid="stChatInput"] textarea {
          background: transparent !important; border: none !important;
        }

        /* ─────────── METRICS ─────────── */
        div[data-testid="stMetric"] {
          background: var(--s1);
          border: 1px solid rgba(255,255,255,.07);
          border-radius: var(--rxl); padding: 1rem;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
          font-family: var(--f); font-size: .68rem; font-weight: 700;
          letter-spacing: .09em; text-transform: uppercase;
          color: var(--m1) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
          font-family: var(--mono); font-weight: 600; color: #fff !important;
        }

        /* ─────────── DATA FRAME ─────────── */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
          border-radius: var(--rxl); overflow: hidden;
          border: 1px solid rgba(255,255,255,.07);
        }

        /* ─────────── EXPANDER ─────────── */
        div[data-testid="stExpander"] {
          background: var(--s2) !important;
          border: 1px solid rgba(255,255,255,.07) !important;
          border-radius: var(--rl) !important;
        }
        div[data-testid="stExpander"] summary {
          font-family: var(--f); font-size: .81rem; font-weight: 600;
          color: var(--m1) !important;
        }
        div[data-testid="stExpander"] summary:hover { color: var(--txt) !important; }

        /* ─────────── CONTAINERS WITH BORDER ─────────── */
        div[data-testid="stVerticalBlockBorderWrapper"] {
          border: 1px solid rgba(255,255,255,.07) !important;
          border-radius: var(--rxl) !important;
          background: var(--s1) !important;
        }

        /* ─────────── ALERTS / CODE ─────────── */
        div[data-testid="stAlert"] {
          border-radius: var(--rl) !important;
          border: 1px solid rgba(255,255,255,.07) !important;
        }
        .stCode, pre {
          border-radius: var(--rl) !important;
          border: 1px solid rgba(255,255,255,.07) !important;
        }

        /* ─────────── SPINNER ─────────── */
        .stSpinner > div { border-top-color: var(--brand) !important; }

        /* ─────────── HIDE CHROME ─────────── */
        footer, #MainMenu, [data-testid="stToolbar"] { visibility: hidden; }

        /* ─────────── SCROLLBAR ─────────── */
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,.10); border-radius: 999px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,.20); }

        /* ─────────── SELECT SLIDER ─────────── */
        .stSelectSlider [data-baseweb="slider"] [role="slider"] {
          background: var(--brand) !important;
        }

        /* ─────────── DIVIDER ─────────── */
        hr { border-color: rgba(255,255,255,.07) !important; margin: .65rem 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── HELPERS ────────────────────────────

def hero(title, copy, eyebrow="Zain 360 Copilot"):
    logo_html = ""
    if LOGO_DATA_URI:
        logo_html = f'<div class="hero-logo-wrap"><img src="{LOGO_DATA_URI}" alt="Zain Logo"></div>'
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-eyebrow">◆ {eyebrow}</div>
          <div class="hero-title">{title}</div>
          <p class="hero-copy">{copy}</p>
          {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, note="", variant="neu"):
    st.markdown(
        f"""
        <div class="kpi {variant}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card_colored(label, value, note="", color=""):
    """KPI card where the value text is colored (ok / warn / bad)."""
    st.markdown(
        f"""
        <div class="kpi {color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {color}">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, sub=""):
    sub_html = f'<span>{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="section-hd"><h3>{title}</h3>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def format_num(value, suffix=""):
    try:
        if pd.isna(value):
            return "0"
        value = float(value)
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M{suffix}"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K{suffix}"
        if value == int(value):
            return f"{int(value):,}{suffix}"
        return f"{value:,.2f}{suffix}"
    except Exception:
        return str(value)


def plotly_layout(fig, height=400, legend=True):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=48, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#8591AB"),
        title_font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color="#E8EBF6"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ) if legend else None,
        xaxis=dict(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.06)"),
        colorway=["#63CEF8", "#8BDFFF", "#7C72D8", "#A935A7", "#17B978", "#E8960F"],
    )
    return fig


def build_chart(df, chart_type, title, x="label", y="value", color=None, height=390):
    if df is None or df.empty:
        st.info("No data available for this visual.")
        return
    common = dict(
        template="plotly_dark", height=height, title=title,
        color_discrete_sequence=["#63CEF8", "#8BDFFF", "#7C72D8", "#A935A7", "#17B978", "#E8960F"],
    )
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
        st.warning(chart.get("summary") or "No matching data was found for this chart request.")
        return
    df = pd.DataFrame(rows)
    section_header(chart.get("title", "Chart"), chart.get("metric", ""))
    build_chart(df, chart.get("chart_type", "bar"), chart.get("title", "Chart"), x="label", y="value")
    if chart.get("summary"):
        st.caption(chart["summary"])
    with st.expander("View chart data"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Export chart data",
            df.to_csv(index=False).encode("utf-8"),
            file_name="chart_data.csv",
            mime="text/csv",
            use_container_width=True,
        )


def build_chat_history_context(chat, limit=8):
    history = []
    for message in chat.get("messages", [])[-limit:]:
        item = {"role": message.get("role", ""), "content": str(message.get("content", ""))[:4000]}
        if message.get("source"):
            item["source"] = message.get("source", "")
        if message.get("sql"):
            item["sql"] = message.get("sql", "")
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
    chat["messages"].append({"role": "user", "content": prompt, "sql": ""})
    try:
        with st.spinner("Analyzing the database…"):
            payload = call_chat_backend(prompt, chat_history)
        answer = payload.get("answer", "No answer was returned.")
        sql = payload.get("sql", "")
        source = payload.get("source", "SQL Agent")
        matched_question = payload.get("matched_question", "")
        match_score = payload.get("match_score", "")
    except Exception as exc:
        answer = (
            "I could not complete this request. "
            f"Details: {type(exc).__name__}: {exc}. "
            "Please confirm the OPENAI_API_KEY is configured."
        )
        sql, source, matched_question, match_score = "", "Error", "", ""
    chat["messages"].append({
        "role": "assistant", "content": answer, "sql": sql,
        "source": source, "matched_question": matched_question, "match_score": match_score,
    })


def run_sql_callback(key_prefix):
    sql = st.session_state.get(f"{key_prefix}_sql_editor", "").strip()
    try:
        st.session_state[f"{key_prefix}_sql_result"] = execute_sql_query(sql)
        st.session_state[f"{key_prefix}_sql_error"] = ""
    except Exception as exc:
        st.session_state[f"{key_prefix}_sql_result"] = None
        st.session_state[f"{key_prefix}_sql_error"] = f"{type(exc).__name__}: {exc}"


def render_sql_runner(default_sql="", key_prefix="sql_runner"):
    editor_key = f"{key_prefix}_sql_editor"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = default_sql
    st.text_area("SQL", height=180, key=editor_key, help="Only safe read-only SELECT queries are allowed.")
    col_run, col_clr, col_tip = st.columns([1, 1, 3])
    with col_run:
        st.button(
            "▶  Run query", type="primary",
            key=f"{key_prefix}_run_button",
            on_click=run_sql_callback, args=(key_prefix,),
            use_container_width=True,
        )
    with col_clr:
        if st.button("✕  Clear", key=f"{key_prefix}_clear", use_container_width=True):
            st.session_state[f"{key_prefix}_sql_result"] = None
            st.session_state[f"{key_prefix}_sql_error"] = ""
            st.rerun()
    with col_tip:
        st.caption("Only SELECT statements are permitted. DDL / DML operations are blocked.")
    error = st.session_state.get(f"{key_prefix}_sql_error", "")
    result = st.session_state.get(f"{key_prefix}_sql_result")
    if error:
        st.error(f"Query failed: {error}")
    elif result:
        rows = result.get("rows", [])
        st.success(f"✓  Returned {len(rows):,} row(s).")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇  Export as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="sql_result.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Query ran successfully but returned no rows.")
        with st.expander("Executed SQL"):
            st.code(result.get("sql", ""), language="sql")


def build_filtered_analytics(month_start, month_end, cities, segments, risk_levels, service_types):
    where = ["m.summary_month BETWEEN ? AND ?"]
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
    where_sql = " AND ".join(where)
    sql = f"""
        SELECT
            m.summary_month, c.customer_id, c.full_name, c.city, c.governorate,
            c.customer_segment, c.customer_type, c.status AS customer_status,
            s.service_type, ch.churn_score, ch.risk_level, ch.main_risk_reason,
            ch.recommended_action, vs.value_segment, vs.arpu_jod,
            vs.total_revenue_6m_jod, m.total_revenue_jod, m.voice_minutes,
            m.data_used_gb, m.sms_count, m.support_interactions_count,
            m.complaints_count, m.payment_delay_days
        FROM customer_monthly_summary m
        JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments vs ON vs.customer_id = c.customer_id
        LEFT JOIN subscriptions s ON s.subscription_id = m.subscription_id
        WHERE {where_sql}
    """
    df = query_df(sql, tuple(params))
    return df, sql, params


# ─────────────────────────── PAGES ────────────────────────────

def show_chat():
    chat = current_chat()
    hero(
        "Customer 360 Chat",
        "Ask direct business questions in plain English. Your chat sessions are saved during this browser session.",
        "Conversational analytics",
    )

    section_header("Quick prompts", "Start with a common question")
    q_cols = st.columns(4)
    for i, question in enumerate(SUGGESTED_QUESTIONS[:4]):
        with q_cols[i]:
            label = question[:54] + "…" if len(question) > 54 else question
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            if message.get("source"):
                st.caption(f"Answered by: {message['source']}")
            st.markdown(message.get("content", ""))
            if "RAG Agent" in message.get("source", "") and message.get("matched_question"):
                score = message.get("match_score", "")
                score_text = f"  ·  Similarity: {score}" if score != "" else ""
                st.caption(f"Memory match: {message['matched_question']}{score_text}")
            if message.get("sql"):
                with st.expander("View generated SQL"):
                    st.code(message["sql"], language="sql")

    if st.session_state.pending_prompt:
        pending = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
        ask_and_store(pending)
        st.rerun()

    prompt = st.chat_input("Ask about churn, customers, revenue, billing, campaigns, complaints, or network impact…")
    if prompt:
        ask_and_store(prompt)
        st.rerun()


def show_dynamic_analytics():
    hero(
        "Dynamic Analytics",
        "Adjust filters, explore KPIs, and drill into charts by month, segment, risk, city, and service type.",
        "Interactive BI",
    )
    options = filter_options()
    months = options["months"]
    if not months:
        st.error("No monthly summary data is available.")
        return

    with st.container(border=True):
        section_header("Filters", "Scope the dataset")
        f1, f2, f3, f4, f5, f6 = st.columns([1.4, 1.2, 1.2, 1.2, 1.2, 1])
        with f1:
            month_start, month_end = st.select_slider(
                "Month range", options=months, value=(months[0], months[-1])
            )
        with f2:
            cities = st.multiselect("Cities", options["cities"], default=[])
        with f3:
            segments = st.multiselect("Segments", options["segments"], default=[])
        with f4:
            risk_levels = st.multiselect("Risk levels", options["risk_levels"], default=[])
        with f5:
            service_types = st.multiselect("Service types", options["service_types"], default=[])
        with f6:
            chart_label = st.selectbox("Chart style", list(CHART_TYPES.keys()), index=0)

        qb1, qb2, qb3, qb4 = st.columns(4)
        with qb1:
            if st.button("⚠  High-risk only", use_container_width=True):
                risk_levels = ["High"]
        with qb2:
            if st.button("★  VIP customers", use_container_width=True):
                segments = ["VIP"] if "VIP" in options["segments"] else segments
        with qb3:
            if st.button("📍 Amman view", use_container_width=True):
                cities = ["Amman"] if "Amman" in options["cities"] else cities
        with qb4:
            if st.button("↺  Reset all filters", use_container_width=True):
                cities, segments, risk_levels, service_types = [], [], [], []

    df, sql, params = build_filtered_analytics(month_start, month_end, cities, segments, risk_levels, service_types)
    if df.empty:
        st.warning("No records match the selected filters.")
        with st.expander("SQL used"):
            st.code(sql, language="sql")
            st.json({"params": params})
        return

    unique_customers = df["customer_id"].nunique()
    total_revenue    = df["total_revenue_jod"].sum()
    avg_churn        = df["churn_score"].mean()
    avg_data         = df["data_used_gb"].mean()
    complaints       = df["complaints_count"].sum()
    support          = df["support_interactions_count"].sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: kpi_card("Customers",   format_num(unique_customers),        "Distinct filtered")
    with k2: kpi_card("Revenue",     format_num(total_revenue, " JOD"),   "Filtered monthly",   "ok")
    with k3: kpi_card("Avg Churn",   f"{avg_churn:.2f}",                  "Average score",      "warn")
    with k4: kpi_card("Avg Data",    format_num(avg_data, " GB"),          "Monthly usage")
    with k5: kpi_card("Complaints",  format_num(complaints),               "In selected range",  "bad")
    with k6: kpi_card("Support",     format_num(support),                  "Interactions")

    chart_type = CHART_TYPES[chart_label]
    chart_tabs = st.tabs(["📈 Revenue", "🔴 Risk", "🏷 Segments", "🗺 Cities", "📶 Usage", "👥 Customers"])

    with chart_tabs[0]:
        monthly = (
            df.groupby("summary_month", as_index=False)["total_revenue_jod"]
            .sum()
            .rename(columns={"summary_month": "label", "total_revenue_jod": "value"})
        )
        build_chart(monthly, "area" if chart_type in {"pie", "doughnut"} else chart_type, "Revenue trend by month")

    with chart_tabs[1]:
        risk = (
            df.drop_duplicates("customer_id")
            .groupby("risk_level", as_index=False)["customer_id"]
            .count()
            .rename(columns={"risk_level": "label", "customer_id": "value"})
        )
        build_chart(risk, "doughnut" if chart_type in {"line", "area"} else chart_type, "Customers by churn risk")

    with chart_tabs[2]:
        seg = (
            df.groupby("customer_segment", as_index=False)["total_revenue_jod"]
            .sum()
            .rename(columns={"customer_segment": "label", "total_revenue_jod": "value"})
            .sort_values("value", ascending=False)
        )
        build_chart(seg, "horizontal_bar" if chart_type in {"pie", "doughnut"} else chart_type, "Revenue by segment")

    with chart_tabs[3]:
        city = (
            df.drop_duplicates("customer_id")
            .groupby("city", as_index=False)["customer_id"]
            .count()
            .rename(columns={"city": "label", "customer_id": "value"})
            .sort_values("value", ascending=False)
            .head(12)
        )
        build_chart(city, "horizontal_bar", "Top cities by customer count")

    with chart_tabs[4]:
        usage = (
            df.groupby("summary_month", as_index=False)[["data_used_gb", "voice_minutes", "sms_count"]]
            .mean()
            .melt(id_vars="summary_month", var_name="metric", value_name="value")
        )
        fig = px.line(
            usage, x="summary_month", y="value", color="metric", markers=True,
            title="Average usage trend", template="plotly_dark",
            color_discrete_sequence=["#63CEF8", "#7C72D8", "#A935A7"],
        )
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    with chart_tabs[5]:
        customer_view = (
            df.groupby(
                ["customer_id", "full_name", "city", "customer_segment", "risk_level", "main_risk_reason"],
                as_index=False,
            ).agg(
                total_revenue_jod=("total_revenue_jod", "sum"),
                avg_churn_score=("churn_score", "mean"),
                complaints=("complaints_count", "sum"),
                support_interactions=("support_interactions_count", "sum"),
                payment_delay_days=("payment_delay_days", "max"),
            ).sort_values(["avg_churn_score", "total_revenue_jod"], ascending=[False, False])
        )
        st.dataframe(customer_view, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇  Export filtered customers",
            customer_view.to_csv(index=False).encode("utf-8"),
            file_name="filtered_customers.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Show SQL + parameters"):
        st.code(sql, language="sql")
        st.json({"params": params})


def show_chart_builder():
    hero(
        "Chart Builder",
        "Describe the chart you want in plain language. The app writes a safe query and renders the result instantly.",
        "Natural-language visuals",
    )
    with st.container(border=True):
        question = st.text_area(
            "Chart description",
            value="Build a chart based on customer with ID = 9 by their complaints type and number.",
            height=110,
        )
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            chart_label = st.selectbox("Chart type", list(CHART_TYPES.keys()))
        with c2:
            run = st.button("▶  Build chart", type="primary", use_container_width=True)
        with c3:
            st.caption("Tip: describe one clear metric — e.g. 'churn by city', 'conversion by campaign', 'complaints by category'.")

    if run:
        with st.spinner("Querying database and building chart…"):
            st.session_state.last_chart = build_chart_from_question(question, CHART_TYPES[chart_label])

    if st.session_state.last_chart:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_chart(st.session_state.last_chart)
        st.markdown("</div>", unsafe_allow_html=True)


def show_sql_workspace():
    hero(
        "SQL Workspace",
        "A clean read-only SQL environment for analysts. Only SELECT statements are permitted — no schema mutations.",
        "Safe query runner",
    )
    templates = {
        "Total customers": "SELECT COUNT(*) AS total_customers FROM customers",
        "Top churn customers": (
            "SELECT c.customer_id, c.full_name, c.city, c.customer_segment,\n"
            "       ch.churn_score, ch.risk_level, ch.main_risk_reason\n"
            "FROM customer_churn_scores ch\n"
            "JOIN customers c ON c.customer_id = ch.customer_id\n"
            "ORDER BY ch.churn_score DESC\n"
            "LIMIT 10"
        ),
        "Revenue by segment": (
            "SELECT vs.value_segment,\n"
            "       COUNT(*) AS customers,\n"
            "       ROUND(AVG(vs.arpu_jod), 2) AS avg_arpu,\n"
            "       ROUND(SUM(vs.total_revenue_6m_jod), 2) AS revenue_6m\n"
            "FROM customer_value_segments vs\n"
            "GROUP BY vs.value_segment\n"
            "ORDER BY revenue_6m DESC"
        ),
        "Open complaints": (
            "SELECT complaint_category, severity, status, COUNT(*) AS total\n"
            "FROM complaints\n"
            "WHERE status != 'Resolved'\n"
            "GROUP BY complaint_category, severity, status\n"
            "ORDER BY total DESC"
        ),
    }
    t_col, b_col = st.columns([2, 1])
    with t_col:
        selected_template = st.selectbox("Query template", list(templates.keys()))
    with b_col:
        st.write("")
        if st.button("⬆  Load template", use_container_width=True):
            st.session_state["standalone_sql_editor"] = templates[selected_template]
            st.rerun()
    render_sql_runner(templates[selected_template], key_prefix="standalone")


def show_suggested_questions():
    hero(
        "Prompt Library",
        "Use ready-made business prompts to generate database-backed answers in seconds.",
        "Suggested workflows",
    )
    cols = st.columns(3)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="prompt-card">
                  <div class="prompt-num">Use case {i + 1:02d}</div>
                  <p>{question}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Send to AI Chat →", key=f"suggested_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.session_state.page = "Chat"
                st.rerun()


def show_customer_insights():
    hero(
        "Customer Insights",
        "Search for any customer and view their full 360° profile — churn risk, value, billing, complaints, support, and usage.",
        "Single customer view",
    )
    search = st.text_input("Search customer", placeholder="Enter customer ID, name, phone, city, or email")
    if search.strip():
        like = f"%{search.strip()}%"
        candidates = query_df(
            """SELECT customer_id, full_name, city, customer_segment, phone_number, email
               FROM customers
               WHERE CAST(customer_id AS TEXT) LIKE ?
                  OR full_name LIKE ?
                  OR phone_number LIKE ?
                  OR email LIKE ?
                  OR city LIKE ?
               ORDER BY customer_id LIMIT 100""",
            (like, like, like, like, like),
        )
    else:
        candidates = query_df(
            "SELECT customer_id, full_name, city, customer_segment, phone_number, email "
            "FROM customers ORDER BY customer_id LIMIT 100"
        )

    if candidates.empty:
        st.warning("No matching customers found.")
        return

    labels = [
        f"{row.customer_id} · {row.full_name} · {row.city} · {row.customer_segment}"
        for row in candidates.itertuples()
    ]
    selected_label = st.selectbox("Select customer", labels)
    customer_id = int(selected_label.split(" · ")[0])

    customer = query_df(
        """SELECT c.*, ch.churn_score, ch.risk_level, ch.main_risk_reason, ch.recommended_action,
                  vs.value_segment, vs.arpu_jod, vs.total_revenue_6m_jod, vs.lifetime_months
           FROM customers c
           LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
           LEFT JOIN customer_value_segments vs ON vs.customer_id = c.customer_id
           WHERE c.customer_id = ?""",
        (customer_id,),
    ).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Customer",     customer["full_name"],                                     f"ID {customer_id}")
    with c2: kpi_card("Risk Level",   customer.get("risk_level", "N/A"),                         f"Score {customer.get('churn_score', 0):.2f}", "bad")
    with c3: kpi_card("Value Segment",customer.get("value_segment", "N/A"),                      format_num(customer.get("arpu_jod", 0), " JOD ARPU"), "ok")
    with c4: kpi_card("6M Revenue",   format_num(customer.get("total_revenue_6m_jod", 0), " JOD"), f"{customer.get('lifetime_months', 0)} months lifetime")

    st.markdown(
        f"""
        <div class="card">
          <div style="margin-bottom:.6rem;">
            <span class="badge">{customer.get("customer_segment", "Segment")}</span>
            <span class="badge">{customer.get("city", "City")}</span>
            <span class="badge">{customer.get("preferred_language", "Language")}</span>
            <span class="badge">{customer.get("customer_status", customer.get("status", "Status"))}</span>
          </div>
          <div style="font-size:.72rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--m2);margin-bottom:.28rem;">Recommended action</div>
          <p style="color:var(--txt);font-size:.89rem;margin:0 0 .85rem;">{customer.get("recommended_action", "No recommended action available.")}</p>
          <div style="font-size:.72rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--m2);margin-bottom:.28rem;">Main risk reason</div>
          <p style="color:var(--m1);font-size:.89rem;margin:0;">{customer.get("main_risk_reason", "No risk reason available.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["👤 Profile", "📡 Subscriptions", "💳 Billing", "⚠ Complaints", "🎧 Support", "📊 Monthly usage", "🤖 Ask AI"])

    with tabs[0]:
        profile_df = customer.to_frame(name="value").reset_index().rename(columns={"index": "field"})
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇  Export profile",
            profile_df.to_csv(index=False).encode("utf-8"),
            file_name=f"customer_{customer_id}_profile.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tabs[1]:
        subs = query_df(
            """SELECT s.subscription_id, s.msisdn, s.service_type, s.activation_date,
                      s.contract_end_date, s.status, s.auto_renewal_flag,
                      s.primary_subscription_flag, p.plan_name, p.plan_category,
                      p.monthly_fee_jod, p.technology
               FROM subscriptions s
               LEFT JOIN plans p ON p.plan_id = s.plan_id
               WHERE s.customer_id = ?
               ORDER BY s.primary_subscription_flag DESC, s.activation_date DESC""",
            (customer_id,),
        )
        st.dataframe(subs, use_container_width=True, hide_index=True)

    with tabs[2]:
        billing = query_df(
            """SELECT i.invoice_id, i.issue_date, i.due_date, i.total_amount_jod,
                      i.payment_status, i.days_overdue, a.account_number, a.account_type
               FROM invoices i
               JOIN accounts a ON a.account_id = i.account_id
               WHERE a.customer_id = ?
               ORDER BY i.issue_date DESC LIMIT 50""",
            (customer_id,),
        )
        st.dataframe(billing, use_container_width=True, hide_index=True)

    with tabs[3]:
        complaints = query_df(
            """SELECT complaint_id, complaint_date, complaint_category, severity, status,
                      resolved_date, compensation_amount_jod, complaint_description
               FROM complaints WHERE customer_id = ?
               ORDER BY complaint_date DESC LIMIT 50""",
            (customer_id,),
        )
        st.dataframe(complaints, use_container_width=True, hide_index=True)

    with tabs[4]:
        support = query_df(
            """SELECT interaction_id, interaction_datetime, channel, reason_category, issue_type,
                      priority, resolution_status, resolution_time_minutes, customer_sentiment
               FROM support_interactions WHERE customer_id = ?
               ORDER BY interaction_datetime DESC LIMIT 50""",
            (customer_id,),
        )
        st.dataframe(support, use_container_width=True, hide_index=True)

    with tabs[5]:
        monthly = query_df(
            """SELECT summary_month, total_revenue_jod, voice_minutes, data_used_gb, sms_count,
                      support_interactions_count, complaints_count, payment_delay_days, churn_score
               FROM customer_monthly_summary WHERE customer_id = ?
               ORDER BY summary_month""",
            (customer_id,),
        )
        if monthly.empty:
            st.info("No monthly usage data available for this customer.")
        else:
            fig = px.line(
                monthly, x="summary_month",
                y=["total_revenue_jod", "data_used_gb", "churn_score"],
                markers=True, title="Customer monthly trend", template="plotly_dark",
                color_discrete_sequence=["#63CEF8", "#7C72D8", "#A935A7"],
            )
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
            st.dataframe(monthly, use_container_width=True, hide_index=True)

    with tabs[6]:
        suggested = f"Show me the full profile, plan, complaints, churn risk, and recommended action for customer {customer_id}."
        st.code(suggested)
        if st.button("Send to AI Chat →", type="primary", use_container_width=True):
            st.session_state.pending_prompt = suggested
            st.session_state.page = "Chat"
            st.rerun()


def show_data_catalog():
    hero(
        "Data Catalog",
        "Browse the SQLite Customer 360 schema, table sizes, and field definitions.",
        "Schema explorer",
    )
    tables = list_tables()
    table_counts = []
    for table in tables:
        try:
            count = query_df(f'SELECT COUNT(*) AS rows FROM "{table}"')["rows"].iloc[0]
        except Exception:
            count = 0
        table_counts.append({"table": table, "rows": count})

    inventory = pd.DataFrame(table_counts).sort_values("rows", ascending=False)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(inventory, use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect table", tables)
    with col2:
        cols_df = table_columns(selected)
        section_header(f"{selected} columns")
        st.dataframe(cols_df[["name", "type", "notnull", "pk"]], use_container_width=True, hide_index=True)
        section_header("Sample rows (10)")
        sample = query_df(f'SELECT * FROM "{selected}" LIMIT 10')
        st.dataframe(sample, use_container_width=True, hide_index=True)

    with st.expander("Export catalog"):
        st.download_button(
            "⬇  Download table inventory",
            inventory.to_csv(index=False).encode("utf-8"),
            file_name="data_catalog.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────── SIDEBAR ────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Brand block
        mark = f'<img src="{LOGO_DATA_URI}" style="width:20px;opacity:.85;">' if LOGO_DATA_URI else "Z"
        st.markdown(
            f"""
            <div class="sb-brand">
              <div class="sb-brand-mark">{mark}</div>
              <div>
                <div class="sb-brand-name">Customer 360</div>
                <div class="sb-brand-sub">AI Copilot · Zain</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # New Chat button
        if st.button("＋  New Chat", type="primary", use_container_width=True):
            create_new_chat()
            st.rerun()

        # Active page indicator
        active = next((item for item in NAV_ITEMS if item[0] == st.session_state.page), NAV_ITEMS[0])
        st.markdown(
            f"""
            <span class="sb-lbl">Current workspace</span>
            <div class="sb-active">
              <span style="font-size:1rem;">{active[2]}</span>
              <div style="flex:1">
                <div class="sb-active-name">{active[1]}</div>
                <div class="sb-active-desc">{active[3]}</div>
              </div>
              <div class="sb-dot"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Recent chats
        st.markdown('<span class="sb-lbl">Saved chats</span>', unsafe_allow_html=True)
        for chat in st.session_state.chat_sessions[:8]:
            label = "💬  " + chat["title"]
            if st.button(label, key=f"select_{chat['id']}", use_container_width=True):
                st.session_state.current_chat_id = chat["id"]
                st.session_state.page = "Chat"
                st.rerun()

        if st.button("✕  Delete current chat", key="del_chat", use_container_width=True):
            delete_current_chat()
            st.rerun()

        # Navigation
        st.markdown('<span class="sb-lbl">Navigation</span>', unsafe_allow_html=True)
        for page, title, icon, _desc in NAV_ITEMS:
            if st.button(f"{icon}  {title}", key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()


# ─────────────────────────── MAIN ────────────────────────────

def main():
    ensure_state()
    inject_css()
    render_sidebar()

    page = st.session_state.page
    if page == "Chat":
        show_chat()
    elif page == "Analytics":
        show_dynamic_analytics()
    elif page == "Chart Builder":
        show_chart_builder()
    elif page == "SQL Query Builder":
        show_sql_workspace()
    elif page == "Suggested Questions":
        show_suggested_questions()
    elif page == "Customer Insights":
        show_customer_insights()
    elif page == "Data Catalog":
        show_data_catalog()
    else:
        show_chat()


if __name__ == "__main__":
    main()
