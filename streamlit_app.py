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
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "ظ‹ع؛â€œظ¹",
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
    ("Chat", "AI Chat", "ظ‹ع؛â€™آ¬", "Ask business questions"),
    ("Analytics", "Dynamic Analytics", "ظ‹ع؛â€œظ¹", "Filter KPIs and charts"),
    ("Chart Builder", "Chart Builder", "ظ‹ع؛â€œث†", "Create custom visuals"),
    ("SQL Query Builder", "SQL Workspace", "ظ‹ع؛آ§آ®", "Run safe SELECT queries"),
    ("Suggested Questions", "Prompt Library", "أ¢إ“آ¨", "Ready-made use cases"),
]


def ensure_state():
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"
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


# أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ CSS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬

def inject_css():
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

          :root {
            --accent:       #64CBF5;
            --accent-dim:   rgba(100,203,245,.18);
            --accent-glow:  rgba(100,203,245,.35);
            --bg:        #07080D;
            --bg2:       #0C0E16;
            --panel:     #10131C;
            --panel2:    #141824;
            --border:    rgba(255,255,255,.07);
            --border2:   rgba(255,255,255,.12);
            --text:      #EEF0F5;
            --muted:     #8A92A6;
            --soft:      #5C6478;
            --good:      #22D3A0;
            --warn:      #F5BE47;
            --danger:    #F05252;
            --radius:    16px;
            --radius-lg: 22px;
            --radius-xl: 28px;
            --font-head: 'Syne', sans-serif;
            --font-body: 'DM Sans', sans-serif;
          }

          *, *::before, *::after { box-sizing: border-box; }

          html, body, .stApp {
            font-family: var(--font-body);
            background: var(--bg);
            color: var(--text);
          }

          .stApp {
            background:
              radial-gradient(ellipse 800px 600px at -10% -15%, rgba(100,203,245,.14) 0%, transparent 55%),
              radial-gradient(ellipse 600px 500px at 110% 5%, rgba(146,50,167,.12) 0%, transparent 50%),
              var(--bg);
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬ Typography reset أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          h1, h2, h3, h4, h5, h6 {
            font-family: var(--font-head);
            color: var(--text);
            letter-spacing: -.03em;
          }

          p, label, span, li { color: var(--text); }

          a { color: var(--accent); text-decoration: none; }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬ Block container أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .block-container {
            padding: 1.5rem 2.25rem 5rem !important;
            max-width: 1560px !important;
          }

          @media (max-width: 900px) {
            .block-container { padding: 1rem 1rem 5rem !important; }
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ SIDEBAR أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          section[data-testid="stSidebar"] {
            background:
              radial-gradient(ellipse 260px 300px at 50% -10%, rgba(100,203,245,.22) 0%, transparent 60%),
              linear-gradient(180deg, #0A0C14 0%, #07080D 100%) !important;
            border-right: 1px solid rgba(255,255,255,.06) !important;
          }

          section[data-testid="stSidebar"] > div {
            padding: 1.25rem 1rem 2rem !important;
          }

          section[data-testid="stSidebar"] * { color: var(--text) !important; }

          /* Sidebar all buttons */
          [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            min-height: 40px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,.04);
            color: var(--muted) !important;
            font-family: var(--font-body);
            font-weight: 500;
            font-size: .83rem;
            justify-content: flex-start;
            text-align: left;
            padding: .55rem .85rem;
            transition: all .16s ease;
            box-shadow: none;
          }

          [data-testid="stSidebar"] .stButton > button:hover {
            border-color: var(--border2);
            background: rgba(255,255,255,.08);
            color: var(--text) !important;
            transform: none;
          }

          /* New Chat primary button in sidebar */
          [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent), #9232A7) !important;
            border-color: rgba(255,255,255,.14) !important;
            color: #fff !important;
            font-weight: 600;
            font-size: .85rem;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬ Sidebar brand block أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .sb-brand {
            display: flex;
            align-items: center;
            gap: .75rem;
            padding: .85rem .95rem;
            background: rgba(100,203,245,.10);
            border: 1px solid rgba(100,203,245,.22);
            border-radius: var(--radius-lg);
            margin-bottom: 1.25rem;
          }

          .sb-brand-mark {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent), #6F4BD8);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            flex-shrink: 0;
          }

          .sb-brand-text { line-height: 1.2; }

          .sb-brand-title {
            font-family: var(--font-head);
            font-size: .93rem;
            font-weight: 700;
            color: #fff !important;
          }

          .sb-brand-sub {
            font-size: .71rem;
            color: rgba(255,255,255,.5) !important;
          }

          /* Sidebar section labels */
          .sb-label {
            font-family: var(--font-head);
            font-size: .64rem;
            font-weight: 700;
            letter-spacing: .14em;
            text-transform: uppercase;
            color: var(--soft) !important;
            margin: 1.1rem .1rem .4rem;
          }

          /* Active page indicator */
          .sb-active {
            display: flex;
            align-items: center;
            gap: .6rem;
            padding: .62rem .9rem;
            background: rgba(100,203,245,.15);
            border: 1px solid rgba(100,203,245,.30);
            border-radius: 12px;
            margin-bottom: .75rem;
            cursor: default;
          }

          .sb-active-icon {
            font-size: 1rem;
          }

          .sb-active-text { flex: 1; }

          .sb-active-name {
            font-family: var(--font-head);
            font-size: .85rem;
            font-weight: 700;
            color: #fff !important;
          }

          .sb-active-desc {
            font-size: .69rem;
            color: rgba(255,255,255,.5) !important;
          }

          .sb-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 6px var(--accent);
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ HERO BANNER أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border2);
            border-radius: var(--radius-xl);
            padding: 1.5rem 1.75rem;
            background:
              radial-gradient(ellipse 60% 120% at 0% 50%, rgba(100,203,245,.20) 0%, transparent 55%),
              linear-gradient(135deg, var(--panel), var(--panel2));
            margin-bottom: 1.4rem;
          }

          .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.015'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            pointer-events: none;
            opacity: .6;
          }

          .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .28rem .6rem;
            background: rgba(100,203,245,.14);
            border: 1px solid rgba(100,203,245,.28);
            border-radius: 999px;
            color: #7FD8FF !important;
            font-family: var(--font-head);
            font-size: .67rem;
            font-weight: 700;
            letter-spacing: .10em;
            text-transform: uppercase;
            margin-bottom: .7rem;
          }

          .hero-title {
            font-family: var(--font-head);
            font-size: clamp(1.65rem, 2.8vw, 2.6rem);
            font-weight: 800;
            letter-spacing: -.045em;
            line-height: 1.05;
            color: #fff !important;
            margin: 0 0 .5rem;
          }

          .hero-copy {
            font-family: var(--font-body);
            font-size: .92rem;
            color: var(--muted) !important;
            line-height: 1.6;
            max-width: 720px;
            margin: 0;
          }

          .hero-logo-wrap {
            position: absolute;
            right: 1.75rem;
            top: 50%;
            transform: translateY(-50%);
            width: clamp(56px, 7vw, 88px);
            height: clamp(56px, 7vw, 88px);
            background: rgba(255,255,255,.045);
            border: 1px solid var(--border2);
            border-radius: var(--radius);
            display: flex;
            align-items: center;
            justify-content: center;
          }

          .hero-logo-wrap img {
            width: 68%;
            opacity: .7;
          }

          @media (max-width: 680px) {
            .hero-logo-wrap { display: none; }
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ KPI CARDS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            gap: .75rem;
            margin-bottom: 1.25rem;
          }

          .kpi {
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1rem 1.1rem;
            background: var(--panel);
            position: relative;
            overflow: hidden;
            transition: border-color .2s, box-shadow .2s;
          }

          .kpi::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent), transparent);
            opacity: 0;
            transition: opacity .2s;
          }

          .kpi:hover { border-color: var(--border2); }
          .kpi:hover::before { opacity: 1; }

          .kpi-label {
            font-family: var(--font-head);
            font-size: .67rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
            color: var(--muted) !important;
            margin-bottom: .55rem;
          }

          .kpi-value {
            font-family: var(--font-head);
            font-size: clamp(1.35rem, 2.4vw, 1.9rem);
            font-weight: 800;
            letter-spacing: -.04em;
            color: #fff !important;
            line-height: 1;
            margin-bottom: .3rem;
          }

          .kpi-note {
            font-size: .75rem;
            color: var(--soft) !important;
            line-height: 1.35;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ CARDS & SHELLS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.15rem;
            margin-bottom: .85rem;
          }

          .card-sm {
            background: var(--panel2);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: .85rem .95rem;
            margin-bottom: .65rem;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ PROMPT CARDS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .prompt-card {
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: .95rem 1rem;
            background: var(--panel);
            min-height: 118px;
            transition: border-color .18s, background .18s;
            cursor: default;
          }

          .prompt-card:hover {
            border-color: rgba(100,203,245,.28);
            background: var(--panel2);
          }

          .prompt-card-num {
            font-family: var(--font-head);
            font-size: .64rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
            color: var(--accent) !important;
            margin-bottom: .42rem;
          }

          .prompt-card p {
            margin: 0;
            color: var(--muted) !important;
            font-size: .84rem;
            line-height: 1.5;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ QUICK PROMPT CHIPS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .quick-chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: .5rem;
            margin-bottom: 1rem;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ SECTION HEADERS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .section-hd {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
            margin: .25rem 0 .9rem;
          }

          .section-hd h3 {
            font-family: var(--font-head);
            font-size: 1rem;
            font-weight: 700;
            margin: 0;
            color: var(--text) !important;
          }

          .section-hd span {
            font-size: .78rem;
            color: var(--soft) !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ TAGS / BADGES أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .badge {
            display: inline-flex;
            align-items: center;
            padding: .22rem .52rem;
            border-radius: 999px;
            background: var(--accent-dim);
            border: 1px solid rgba(100,203,245,.22);
            color: #8BE0FF !important;
            font-size: .69rem;
            font-weight: 600;
            margin: .1rem .18rem .1rem 0;
          }

          .badge-good {
            background: rgba(34,211,160,.12);
            border-color: rgba(34,211,160,.22);
            color: var(--good) !important;
          }

          .badge-warn {
            background: rgba(245,190,71,.12);
            border-color: rgba(245,190,71,.22);
            color: var(--warn) !important;
          }

          .badge-danger {
            background: rgba(240,82,82,.12);
            border-color: rgba(240,82,82,.22);
            color: var(--danger) !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ DIVIDER أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          hr { border-color: var(--border) !important; margin: .75rem 0 !important; }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ BUTTONS (main content) أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .stButton > button,
          .stDownloadButton > button,
          div[data-testid="stFormSubmitButton"] > button {
            font-family: var(--font-body) !important;
            font-weight: 500 !important;
            min-height: 40px;
            border-radius: 12px !important;
            border: 1px solid var(--border2) !important;
            background: var(--panel2) !important;
            color: var(--text) !important;
            box-shadow: none !important;
            transition: border-color .15s, background .15s, transform .12s !important;
            font-size: .86rem !important;
          }

          .stButton > button:hover,
          .stDownloadButton > button:hover {
            border-color: rgba(100,203,245,.45) !important;
            background: var(--panel) !important;
            transform: translateY(-1px) !important;
          }

          .stButton > button[kind="primary"],
          .stDownloadButton > button[kind="primary"],
          div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, var(--accent), #9232A7) !important;
            border-color: rgba(255,255,255,.14) !important;
            color: #fff !important;
            font-weight: 600 !important;
          }

          .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #7DDCFF, #6F4BD8) !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ INPUTS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          input,
          textarea,
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div {
            font-family: var(--font-body) !important;
            background: var(--panel2) !important;
            border: 1px solid var(--border2) !important;
            border-radius: 12px !important;
            color: var(--text) !important;
          }

          textarea {
            font-family: ui-monospace, 'Fira Code', Menlo, monospace !important;
            font-size: .84rem !important;
          }

          /* Slider */
          .stSlider [data-testid="stTickBar"] { display: none; }
          .stSlider [data-baseweb="slider"] div[role="slider"] {
            background: var(--accent) !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ TABS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          div[data-testid="stTabs"] [role="tablist"] {
            gap: .3rem;
            border-bottom: 1px solid var(--border) !important;
            padding-bottom: 0;
          }

          div[data-testid="stTabs"] button[role="tab"] {
            font-family: var(--font-head) !important;
            font-size: .8rem !important;
            font-weight: 600 !important;
            color: var(--muted) !important;
            border-radius: 8px 8px 0 0 !important;
            padding: .45rem .9rem !important;
            border: 1px solid transparent !important;
            border-bottom: none !important;
            background: transparent !important;
            transition: color .15s, background .15s !important;
          }

          div[data-testid="stTabs"] button[role="tab"]:hover {
            color: var(--text) !important;
            background: var(--panel) !important;
          }

          div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #fff !important;
            background: var(--panel) !important;
            border-color: var(--border) !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ CHAT MESSAGES أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .stChatMessage {
            background: var(--panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            padding: .85rem !important;
          }

          [data-testid="stChatMessageContent"] p {
            font-size: .92rem;
            line-height: 1.65;
          }

          [data-testid="stChatInput"] {
            border-radius: var(--radius-lg) !important;
            background: var(--panel2) !important;
            border: 1px solid var(--border2) !important;
          }

          [data-testid="stChatInput"] textarea {
            background: transparent !important;
            border: none !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ METRICS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1rem;
          }

          div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            font-family: var(--font-head);
            font-size: .7rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: var(--muted) !important;
          }

          div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-family: var(--font-head);
            font-weight: 800;
            color: #fff !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ DATAFRAME أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          div[data-testid="stDataFrame"],
          div[data-testid="stTable"] {
            border-radius: var(--radius-lg);
            overflow: hidden;
            border: 1px solid var(--border);
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ EXPANDER أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          div[data-testid="stExpander"] {
            background: var(--panel2) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
          }

          div[data-testid="stExpander"] summary {
            font-family: var(--font-head);
            font-size: .82rem;
            font-weight: 600;
            color: var(--muted) !important;
          }

          div[data-testid="stExpander"] summary:hover {
            color: var(--text) !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ CONTAINERS WITH BORDER أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            background: var(--panel) !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ ALERTS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          div[data-testid="stAlert"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
            font-family: var(--font-body) !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ CODE أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .stCode, pre {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
          }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ SPINNER أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .stSpinner > div { border-top-color: var(--accent) !important; }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ HIDE DEFAULT STREAMLIT CHROME أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          footer, #MainMenu, [data-testid="stToolbar"] { visibility: hidden; }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ SCROLLBAR أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          ::-webkit-scrollbar { width: 5px; height: 5px; }
          ::-webkit-scrollbar-track { background: transparent; }
          ::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border-radius: 999px; }
          ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,.22); }

          /* أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ SELECT SLIDER OVERRIDE أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ */
          .stSelectSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--accent) !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Store plot template
    st.session_state.plot_template = "plotly_dark"


# أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ HELPERS أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬

def hero(title, copy, eyebrow="Zain 360 Copilot"):
    logo_html = ""
    if LOGO_DATA_URI:
        logo_html = f"""
        <div class="hero-logo-wrap">
          <img src="{LOGO_DATA_URI}" alt="Zain Logo">
        </div>"""
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-eyebrow">أ¢â€”â€  {eyebrow}</div>
          <div class="hero-title">{title}</div>
          <p class="hero-copy">{copy}</p>
          {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
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
            return f"{value/1_000_000:.2f}M{suffix}"
        if abs(value) >= 1_000:
            return f"{value/1_000:.1f}K{suffix}"
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
        font=dict(family="DM Sans, sans-serif", size=12, color="#8A92A6"),
        title_font=dict(family="Syne, sans-serif", size=14, color="#EEF0F5"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ) if legend else None,
        xaxis=dict(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.06)"),
        colorway=["#64CBF5", "#7FD8FF", "#6F4BD8", "#B43AA7", "#22D3A0", "#F5BE47"],
    )
    return fig


def build_chart(df, chart_type, title, x="label", y="value", color=None, height=380):
    if df is None or df.empty:
        st.info("No data available for this visual.")
        return

    common = dict(template="plotly_dark", height=height, title=title,
                  color_discrete_sequence=["#64CBF5", "#7FD8FF", "#6F4BD8", "#B43AA7", "#22D3A0", "#F5BE47"])
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
        item = {
            "role": message.get("role", ""),
            "content": str(message.get("content", ""))[:4000],
        }
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
        with st.spinner("Analyzing the databaseأ¢â‚¬آ¦"):
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
        sql = ""
        source = "Error"
        matched_question = ""
        match_score = ""

    chat["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sql": sql,
            "source": source,
            "matched_question": matched_question,
            "match_score": match_score,
        }
    )


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
            "أ¢â€“آ¶  Run query",
            type="primary",
            key=f"{key_prefix}_run_button",
            on_click=run_sql_callback,
            args=(key_prefix,),
            use_container_width=True,
        )
    with col_clr:
        if st.button("أ¢إ“â€¢  Clear", key=f"{key_prefix}_clear", use_container_width=True):
            st.session_state[f"{key_prefix}_sql_result"] = None
            st.session_state[f"{key_prefix}_sql_error"] = ""
            st.rerun()
    with col_tip:
        st.caption("Only SELECT statements are permitted. DDL/DML operations are blocked.")

    error = st.session_state.get(f"{key_prefix}_sql_error", "")
    result = st.session_state.get(f"{key_prefix}_sql_result")
    if error:
        st.error(f"Query failed: {error}")
    elif result:
        rows = result.get("rows", [])
        st.success(f"أ¢إ“â€œ  Returned {len(rows):,} row(s).")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "أ¢آ¬â€،  Export as CSV",
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


# أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ PAGES أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬

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
            label = question[:52] + "أ¢â‚¬آ¦" if len(question) > 52 else question
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
                score_text = f"  ط¢آ·  Similarity: {score}" if score != "" else ""
                st.caption(f"Memory match: {message['matched_question']}{score_text}")
            if message.get("sql"):
                with st.expander("View generated SQL"):
                    st.code(message["sql"], language="sql")

    if st.session_state.pending_prompt:
        pending = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
        ask_and_store(pending)
        st.rerun()

    prompt = st.chat_input("Ask about churn, customers, revenue, billing, campaigns, complaints, or network impactأ¢â‚¬آ¦")
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
            if st.button("أ¢ع‘آ   High-risk only", use_container_width=True):
                risk_levels = ["High"]
        with qb2:
            if st.button("أ¢ع©â€¦  VIP customers", use_container_width=True):
                segments = ["VIP"] if "VIP" in options["segments"] else segments
        with qb3:
            if st.button("ظ‹ع؛â€œع† Amman view", use_container_width=True):
                cities = ["Amman"] if "Amman" in options["cities"] else cities
        with qb4:
            if st.button("أ¢â€ ط›  Reset all filters", use_container_width=True):
                cities, segments, risk_levels, service_types = [], [], [], []

    df, sql, params = build_filtered_analytics(month_start, month_end, cities, segments, risk_levels, service_types)
    if df.empty:
        st.warning("No records match the selected filters.")
        with st.expander("SQL used"):
            st.code(sql, language="sql")
            st.json({"params": params})
        return

    unique_customers = df["customer_id"].nunique()
    total_revenue = df["total_revenue_jod"].sum()
    avg_churn = df["churn_score"].mean()
    avg_data = df["data_used_gb"].mean()
    complaints = df["complaints_count"].sum()
    support = df["support_interactions_count"].sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: kpi_card("Customers", format_num(unique_customers), "Distinct filtered")
    with k2: kpi_card("Revenue", format_num(total_revenue, " JOD"), "Filtered monthly")
    with k3: kpi_card("Avg Churn", f"{avg_churn:.2f}", "Average score")
    with k4: kpi_card("Avg Data", format_num(avg_data, " GB"), "Monthly usage")
    with k5: kpi_card("Complaints", format_num(complaints), "In selected range")
    with k6: kpi_card("Support", format_num(support), "Interactions")

    chart_type = CHART_TYPES[chart_label]
    chart_tabs = st.tabs(["ظ‹ع؛â€œث† Revenue", "ظ‹ع؛â€‌آ´ Risk", "ظ‹ع؛عˆآ· Segments", "ظ‹ع؛â€”ط› Cities", "ظ‹ع؛â€œآ¶ Usage", "ظ‹ع؛â€کآ¥ Customers"])

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
            title="Average usage trend",
            template="plotly_dark",
            color_discrete_sequence=["#64CBF5", "#6F4BD8", "#B43AA7"],
        )
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    with chart_tabs[5]:
        customer_view = (
            df.groupby(
                ["customer_id", "full_name", "city", "customer_segment", "risk_level", "main_risk_reason"],
                as_index=False,
            )
            .agg(
                total_revenue_jod=("total_revenue_jod", "sum"),
                avg_churn_score=("churn_score", "mean"),
                complaints=("complaints_count", "sum"),
                support_interactions=("support_interactions_count", "sum"),
                payment_delay_days=("payment_delay_days", "max"),
            )
            .sort_values(["avg_churn_score", "total_revenue_jod"], ascending=[False, False])
        )
        st.dataframe(customer_view, use_container_width=True, hide_index=True)
        st.download_button(
            "أ¢آ¬â€،  Export filtered customers",
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
            run = st.button("أ¢â€“آ¶  Build chart", type="primary", use_container_width=True)
        with c3:
            st.caption("Tip: describe one clear metric أ¢â‚¬â€‌ e.g. 'churn by city', 'conversion by campaign', 'complaints by category'.")

    if run:
        with st.spinner("Querying database and building chartأ¢â‚¬آ¦"):
            st.session_state.last_chart = build_chart_from_question(question, CHART_TYPES[chart_label])

    if st.session_state.last_chart:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_chart(st.session_state.last_chart)
        st.markdown("</div>", unsafe_allow_html=True)


def show_sql_workspace():
    hero(
        "SQL Workspace",
        "A clean read-only SQL environment for analysts. Only SELECT statements are permitted أ¢â‚¬â€‌ no schema mutations.",
        "Safe query runner",
    )

    templates = {
        "Total customers": "SELECT COUNT(*) AS total_customers FROM customers",
        "Top churn customers": """SELECT c.customer_id, c.full_name, c.city, c.customer_segment,
       ch.churn_score, ch.risk_level, ch.main_risk_reason
FROM customer_churn_scores ch
JOIN customers c ON c.customer_id = ch.customer_id
ORDER BY ch.churn_score DESC
LIMIT 10""",
        "Revenue by segment": """SELECT vs.value_segment,
       COUNT(*) AS customers,
       ROUND(AVG(vs.arpu_jod), 2) AS avg_arpu,
       ROUND(SUM(vs.total_revenue_6m_jod), 2) AS revenue_6m
FROM customer_value_segments vs
GROUP BY vs.value_segment
ORDER BY revenue_6m DESC""",
        "Open complaints": """SELECT complaint_category, severity, status, COUNT(*) AS total
FROM complaints
WHERE status != 'Resolved'
GROUP BY complaint_category, severity, status
ORDER BY total DESC""",
    }

    t_col, b_col = st.columns([2, 1])
    with t_col:
        selected_template = st.selectbox("Query template", list(templates.keys()))
    with b_col:
        st.write("")
        if st.button("أ¢آ¬â€   Load template", use_container_width=True):
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
                  <div class="prompt-card-num">Use case {i + 1:02d}</div>
                  <p>{question}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Send to AI Chat أ¢â€ â€™", key=f"suggested_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.session_state.page = "Chat"
                st.rerun()


def show_customer_insights():
    hero(
        "Customer Insights",
        "Search for any customer and view their full 360ط¢آ° profile أ¢â‚¬â€‌ churn risk, value, billing, complaints, support, and usage.",
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
            """SELECT customer_id, full_name, city, customer_segment, phone_number, email
               FROM customers ORDER BY customer_id LIMIT 100"""
        )

    if candidates.empty:
        st.warning("No matching customers found.")
        return

    labels = [
        f"{row.customer_id} ط¢آ· {row.full_name} ط¢آ· {row.city} ط¢آ· {row.customer_segment}"
        for row in candidates.itertuples()
    ]
    selected_label = st.selectbox("Select customer", labels)
    customer_id = int(selected_label.split(" ط¢آ· ")[0])

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
    with c1: kpi_card("Customer", customer["full_name"], f"ID {customer_id}")
    with c2: kpi_card("Risk Level", customer.get("risk_level", "N/A"), f"Score {customer.get('churn_score', 0):.2f}")
    with c3: kpi_card("Value Segment", customer.get("value_segment", "N/A"), format_num(customer.get("arpu_jod", 0), " JOD ARPU"))
    with c4: kpi_card("6M Revenue", format_num(customer.get("total_revenue_6m_jod", 0), " JOD"), f"{customer.get('lifetime_months', 0)} months lifetime")

    st.markdown(
        f"""
        <div class="card">
          <div style="margin-bottom:.65rem;">
            <span class="badge">{customer.get("customer_segment", "Segment")}</span>
            <span class="badge">{customer.get("city", "City")}</span>
            <span class="badge">{customer.get("preferred_language", "Language")}</span>
            <span class="badge">{customer.get("customer_status", customer.get("status", "Status"))}</span>
          </div>
          <div style="font-family:var(--font-head);font-size:.75rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:.3rem;">Recommended action</div>
          <p style="color:var(--text);font-size:.9rem;margin:0 0 .9rem;">{customer.get("recommended_action", "No recommended action available.")}</p>
          <div style="font-family:var(--font-head);font-size:.75rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:.3rem;">Main risk reason</div>
          <p style="color:var(--muted);font-size:.9rem;margin:0;">{customer.get("main_risk_reason", "No risk reason available.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["ظ‹ع؛â€کآ¤ Profile", "ظ‹ع؛â€œطŒ Subscriptions", "ظ‹ع؛â€™آ³ Billing", "أ¢ع‘آ  Complaints", "ظ‹ع؛عکآ§ Support", "ظ‹ع؛â€œظ¹ Monthly usage", "ظ‹ع؛آ¤â€“ Ask AI"])

    with tabs[0]:
        profile_df = customer.to_frame(name="value").reset_index().rename(columns={"index": "field"})
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        st.download_button(
            "أ¢آ¬â€،  Export profile",
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
                monthly,
                x="summary_month",
                y=["total_revenue_jod", "data_used_gb", "churn_score"],
                markers=True,
                title="Customer monthly trend",
                template="plotly_dark",
                color_discrete_sequence=["#64CBF5", "#6F4BD8", "#B43AA7"],
            )
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
            st.dataframe(monthly, use_container_width=True, hide_index=True)

    with tabs[6]:
        suggested = f"Show me the full profile, plan, complaints, churn risk, and recommended action for customer {customer_id}."
        st.code(suggested)
        if st.button("Send to AI Chat أ¢â€ â€™", type="primary", use_container_width=True):
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
            "أ¢آ¬â€،  Download table inventory",
            inventory.to_csv(index=False).encode("utf-8"),
            file_name="data_catalog.csv",
            mime="text/csv",
            use_container_width=True,
        )


# أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ SIDEBAR أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬

def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sb-brand">
              <div class="sb-brand-mark">{'<img src="' + LOGO_DATA_URI + '" style="width:22px;opacity:.85;">' if LOGO_DATA_URI else 'ظ‹ع؛â€œظ¹'}</div>
              <div class="sb-brand-text">
                <div class="sb-brand-title">Customer 360</div>
                <div class="sb-brand-sub">AI Copilot ط¢آ· Zain</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("أ¯آ¼â€¹  New Chat", type="primary", use_container_width=True):
            create_new_chat()
            st.rerun()

        # Active page indicator
        active_item = next((item for item in NAV_ITEMS if item[0] == st.session_state.page), NAV_ITEMS[0])
        st.markdown(
            f"""
            <div class="sb-label">Current workspace</div>
            <div class="sb-active">
              <div class="sb-active-icon">{active_item[2]}</div>
              <div class="sb-active-text">
                <div class="sb-active-name">{active_item[1]}</div>
                <div class="sb-active-desc">{active_item[3]}</div>
              </div>
              <div class="sb-dot"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-label">Saved chats</div>', unsafe_allow_html=True)
        for chat in st.session_state.chat_sessions[:8]:
            label = "ظ‹ع؛â€™آ¬  " + chat["title"]
            if st.button(label, key=f"select_{chat['id']}", use_container_width=True):
                st.session_state.current_chat_id = chat["id"]
                st.session_state.page = "Chat"
                st.rerun()

        if st.button("أ¢إ“â€¢  Delete current chat", key="del_chat", use_container_width=True):
            delete_current_chat()
            st.rerun()

        st.markdown('<div class="sb-label">Navigation</div>', unsafe_allow_html=True)
        for page, title, icon, _desc in NAV_ITEMS:
            if st.button(f"{icon}  {title}", key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()


# أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬ MAIN أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬أ¢â€‌â‚¬

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
    else:
        show_chat()


if __name__ == "__main__":
    main()
