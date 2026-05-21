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
backend               = importlib.reload(backend)
ask_sql_agent_payload = backend.ask_sql_agent_payload
build_chart_from_question = backend.build_chart_from_question
execute_sql_query     = backend.execute_sql_query
get_database_overview = backend.get_database_overview

_icon = str(LOGO_PATH) if LOGO_PATH.exists() else "🅩"
st.set_page_config(
    page_title="Zain Customer 360 Copilot",
    page_icon=_icon,
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
    ("Chat",              "AI Chat",            "💬", "Ask business questions"),
    ("Analytics",         "Dynamic Analytics",  "📊", "Filter KPIs and charts"),
    ("Chart Builder",     "Chart Builder",      "📈", "Create custom visuals"),
    ("SQL Query Builder", "SQL Workspace",      "🧮", "Run safe SELECT queries"),
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
            "id": "chat_1",
            "title": "New Chat",
            "messages": [default_assistant_message()],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }]
        st.session_state.current_chat_id = "chat_1"
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
    if "last_chart" not in st.session_state:
        st.session_state.last_chart = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    # NEW: toast queue, copy-click flag, search query
    if "toasts" not in st.session_state:
        st.session_state.toasts = []
    if "chat_search" not in st.session_state:
        st.session_state.chat_search = ""


def default_assistant_message():
    return {
        "role": "assistant",
        "content": (
            "Hello. I am your Customer 360 AI Copilot. "
            "Ask me about customers, churn, complaints, billing, campaigns, "
            "support interactions, network events, or revenue performance."
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
    next_id = f"chat_{len(st.session_state.chat_sessions)+1}_{int(time.time())}"
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
#  CSS  (ENHANCED REVAMP)
# ─────────────────────────────────────────────────────────────
def inject_css():
    dark = st.session_state.get("theme_mode", "Dark") == "Dark"

    if dark:
        p = dict(
            bg="#07090f", bg2="#0d1017",
            surf="rgba(20,24,36,.95)", surf2="rgba(26,31,46,.90)",
            bdr="rgba(255,255,255,.08)", bdr2="rgba(255,255,255,.14)",
            txt="#f0f2f8", muted="#8892a4", soft="#5a6478",
            accent="#d71920", accent2="#ff3e47",
            good="#1cc98a", warn="#f5a623",
            shadow="rgba(0,0,0,.45)", input="rgba(26,31,46,.95)",
            plot_template="plotly_dark",
            sidebar_bg="linear-gradient(180deg,rgba(11,13,22,.99),rgba(7,9,16,1))",
            sidebar_glow="rgba(215,25,32,.26)",
        )
    else:
        p = dict(
            bg="#f4f6fb", bg2="#ffffff",
            surf="rgba(255,255,255,.97)", surf2="rgba(247,249,253,.98)",
            bdr="rgba(13,18,30,.09)", bdr2="rgba(13,18,30,.16)",
            txt="#111520", muted="#5a6478", soft="#8892a4",
            accent="#d71920", accent2="#b01218",
            good="#0d8a5e", warn="#a06000",
            shadow="rgba(23,31,56,.10)", input="rgba(255,255,255,.98)",
            plot_template="plotly_white",
            sidebar_bg="linear-gradient(180deg,rgba(11,13,22,.99),rgba(7,9,16,1))",
            sidebar_glow="rgba(215,25,32,.26)",
        )

    st.session_state.plot_template = p["plot_template"]

    st.markdown(f"""
<style>
/* ═══════════════════════════
   ZAIN 360  — ENHANCED UI v2
   ═══════════════════════════ */
:root {{
  --bg:{p['bg']}; --bg2:{p['bg2']};
  --surf:{p['surf']}; --surf2:{p['surf2']};
  --bdr:{p['bdr']}; --bdr2:{p['bdr2']};
  --txt:{p['txt']}; --muted:{p['muted']}; --soft:{p['soft']};
  --accent:{p['accent']}; --accent2:{p['accent2']};
  --good:{p['good']}; --warn:{p['warn']};
  --shadow:{p['shadow']}; --input:{p['input']};
  --r28:28px; --r24:24px; --r20:20px; --r16:16px; --r12:12px; --r8:8px;
}}

/* ── APP BG ── */
.stApp {{
  background:
    radial-gradient(circle at 0% -8%,  rgba(215,25,32,.20),transparent 30%),
    radial-gradient(circle at 96% 0%,  rgba(140,20,28,.10),transparent 22%),
    linear-gradient(160deg,var(--bg),var(--bg2));
  color:var(--txt);
}}
.block-container{{padding:1.4rem 2rem 5rem;max-width:1480px}}
@media(max-width:860px){{.block-container{{padding:1rem 1rem 5.5rem}}}}
h1,h2,h3,h4,h5,h6,p,label,span{{color:var(--txt)}}
h1{{font-size:clamp(1.9rem,3vw,3.2rem);font-weight:900;letter-spacing:-.055em;line-height:1.02}}
h2{{font-weight:800;letter-spacing:-.03em}}
h3{{font-weight:800;letter-spacing:-.02em}}
a{{color:var(--accent2)}}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"]{{
  background:
    radial-gradient(circle at 18% 4%,{p['sidebar_glow']},transparent 32%),
    {p['sidebar_bg']} !important;
  border-right:1px solid rgba(255,255,255,.09);
}}
section[data-testid="stSidebar"] *{{color:#f0f2f8}}
section[data-testid="stSidebar"] > div{{padding:1rem .85rem 1.5rem}}

/* NEW: sidebar search box */
.sidebar-search {{
  display:flex;align-items:center;gap:8px;
  background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.10);
  border-radius:var(--r12);
  padding:7px 11px;margin-bottom:6px;
}}
.sidebar-search input{{
  background:transparent;border:none;outline:none;
  color:#f0f2f8;font-size:.82rem;width:100%;
}}
.sidebar-search input::placeholder{{color:rgba(240,242,248,.35)}}
.sidebar-search-icon{{font-size:13px;opacity:.45;flex-shrink:0}}

[data-testid="stSidebar"] .stButton > button{{
  width:100%;min-height:46px;border-radius:var(--r12);
  border:1px solid rgba(255,255,255,.09);
  background:rgba(255,255,255,.05);color:#f0f2f8;
  font-weight:750;justify-content:flex-start;text-align:left;
  padding:.7rem .9rem;box-shadow:none;transition:all .17s ease;font-size:.88rem;
}}
[data-testid="stSidebar"] .stButton > button:hover{{
  transform:translateY(-1px);
  border-color:rgba(255,255,255,.20);
  background:rgba(255,255,255,.09);
}}

/* ── BRAND CARD ── */
.brand-card{{
  position:relative;overflow:hidden;
  border:1px solid rgba(215,25,32,.30);border-radius:var(--r20);
  padding:1rem 1rem 1.1rem;
  background:linear-gradient(135deg,rgba(215,25,32,.22),rgba(255,255,255,.04)),rgba(255,255,255,.05);
  box-shadow:0 0 0 4px rgba(215,25,32,.07);margin:.2rem 0 .9rem;
}}
.brand-card-inner{{display:flex;align-items:center;gap:10px;margin-bottom:.55rem}}
.brand-logo-img{{
  width:42px;height:42px;border-radius:12px;
  object-fit:cover;flex-shrink:0;
  box-shadow:0 4px 14px rgba(0,0,0,.35),0 0 0 2px rgba(255,255,255,.12);
}}
.brand-title{{font-size:.98rem;font-weight:900;letter-spacing:-.025em;margin-bottom:.4rem}}
.brand-copy{{font-size:.74rem;line-height:1.55;color:rgba(240,242,248,.60)}}
.chip-row{{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.85rem}}
.chip{{
  border:1px solid rgba(255,255,255,.13);border-radius:999px;
  padding:.24rem .55rem;font-size:.65rem;font-weight:850;
  color:#fff;background:rgba(255,255,255,.07);
}}

/* ── SIDEBAR LABELS & ACTIVE ── */
.side-label{{
  margin:.9rem .1rem .4rem;color:rgba(240,242,248,.38);
  font-size:.66rem;text-transform:uppercase;letter-spacing:.12em;font-weight:900;
}}
.active-page{{
  border:1px solid rgba(215,25,32,.50);
  background:linear-gradient(135deg,rgba(215,25,32,.22),rgba(255,255,255,.07));
  border-radius:var(--r16);padding:.72rem .88rem;
  margin:.2rem 0 .55rem;color:#fff;
  box-shadow:0 0 0 4px rgba(215,25,32,.10);font-weight:900;
}}
.active-page small{{display:block;color:rgba(247,248,250,.60);font-size:.68rem;font-weight:600;margin-top:.12rem}}

/* ── HERO CARD ── */
.hero-card{{
  position:relative;overflow:hidden;
  border:1px solid var(--bdr2);border-radius:var(--r28);
  padding:1.4rem min(12rem,18vw) 1.4rem 1.5rem;
  background:
    radial-gradient(circle at 8% -10%,rgba(215,25,32,.22),transparent 30%),
    linear-gradient(135deg,var(--surf),var(--surf2));
  box-shadow:0 24px 80px var(--shadow);margin-bottom:1.2rem;
}}
.hero-card::after{{
  content:"";position:absolute;right:1.4rem;top:50%;
  width:clamp(76px,10vw,130px);height:clamp(76px,10vw,130px);
  transform:translateY(-50%);border-radius:var(--r24);
  background:transparent;border:none;
}}
.hero-logo{{
  position:absolute;right:1.6rem;top:50%;
  width:clamp(64px,8vw,108px);height:clamp(64px,8vw,108px);
  transform:translateY(-50%);
  border-radius:28px;
  object-fit:cover;
  box-shadow:0 8px 32px rgba(0,0,0,.35),0 0 0 3px rgba(255,255,255,.10);
  z-index:1;pointer-events:none;
}}
.hero-eyebrow{{
  display:inline-flex;align-items:center;gap:.45rem;
  padding:.3rem .65rem;border:1px solid var(--bdr2);border-radius:999px;
  background:rgba(215,25,32,.10);color:var(--accent2);
  font-size:.70rem;font-weight:900;letter-spacing:.09em;text-transform:uppercase;margin-bottom:.85rem;
}}
.hero-title{{
  font-size:clamp(1.8rem,3vw,3.1rem);font-weight:950;
  letter-spacing:-.06em;line-height:1.02;color:var(--txt);max-width:900px;
}}
.hero-copy{{color:var(--muted);font-size:.96rem;line-height:1.6;max-width:860px;margin-top:.65rem}}
@media(max-width:700px){{
  .hero-card{{padding:1.2rem}}
  .hero-card::after,.hero-logo{{display:none}}
}}

/* ── KPI CARDS ── */
.kpi-card{{
  border:1px solid var(--bdr);border-radius:var(--r24);padding:1.05rem;min-height:120px;
  background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.015)),var(--surf);
  box-shadow:0 18px 46px var(--shadow);
  transition:transform .18s ease,border-color .18s ease;
}}
.kpi-card:hover{{transform:translateY(-2px);border-color:var(--bdr2)}}
.kpi-label{{color:var(--muted);font-size:.76rem;font-weight:800;text-transform:uppercase;letter-spacing:.07em}}
.kpi-value{{
  color:var(--txt);font-size:clamp(1.4rem,2.4vw,2rem);
  font-weight:950;line-height:1;margin:.55rem 0 .3rem;letter-spacing:-.04em;
}}
.kpi-sub{{color:var(--soft);font-size:.75rem;line-height:1.4}}

/* NEW: trend badge on KPI */
.kpi-trend{{
  display:inline-flex;align-items:center;gap:3px;
  font-size:.70rem;font-weight:800;padding:2px 8px;
  border-radius:999px;margin-top:.45rem;
}}
.kpi-trend.up{{background:rgba(28,201,138,.12);border:1px solid rgba(28,201,138,.25);color:var(--good)}}
.kpi-trend.down{{background:rgba(215,25,32,.10);border:1px solid rgba(215,25,32,.22);color:var(--accent2)}}
.kpi-trend.flat{{background:rgba(255,255,255,.07);border:1px solid var(--bdr2);color:var(--muted)}}

/* ── SECTION TITLES ── */
.section-title{{
  display:flex;align-items:center;justify-content:space-between;
  gap:1rem;margin:.7rem 0 .7rem;
}}
.section-title h3{{margin:0;font-size:1.12rem;font-weight:900}}
.section-title span{{color:var(--muted);font-size:.82rem}}

/* ── PROMPT CARDS ── */
.prompt-card{{
  border:1px solid var(--bdr);border-radius:var(--r20);padding:.9rem;
  background:var(--surf);box-shadow:0 14px 36px var(--shadow);
  min-height:120px;cursor:pointer;
  transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;
}}
.prompt-card:hover{{
  transform:translateY(-3px);
  border-color:rgba(215,25,32,.40);
  box-shadow:0 0 0 3px rgba(215,25,32,.08),0 22px 55px var(--shadow);
}}
.prompt-card b{{display:block;margin-bottom:.35rem}}
.prompt-card p{{margin:0;color:var(--muted);font-size:.84rem;line-height:1.45}}

/* NEW: message source badge */
.msg-source-badge{{
  display:inline-flex;align-items:center;gap:4px;
  margin-bottom:5px;padding:2px 8px;border-radius:999px;
  font-size:.67rem;font-weight:800;letter-spacing:.07em;text-transform:uppercase;
  background:rgba(215,25,32,.10);border:1px solid rgba(215,25,32,.22);color:var(--accent2);
}}
.msg-source-badge.rag{{
  background:rgba(28,201,138,.09);border-color:rgba(28,201,138,.22);color:var(--good);
}}
.msg-source-badge.error{{
  background:rgba(215,25,32,.14);border-color:rgba(215,25,32,.35);color:var(--accent2);
}}

/* NEW: message timestamp */
.msg-ts{{font-size:.65rem;color:var(--soft);margin-top:4px;text-align:right}}
.msg-ts.bot{{text-align:left}}

/* ── CHAT MESSAGES ── */
.stChatMessage{{
  border-radius:var(--r24);border:1px solid var(--bdr);
  background:var(--surf);box-shadow:0 10px 30px var(--shadow);padding:.8rem;
}}

/* NEW: chat input glow on focus */
[data-testid="stChatInput"] textarea:focus{{
  border-color:rgba(215,25,32,.50) !important;
  box-shadow:0 0 0 3px rgba(215,25,32,.10) !important;
}}
[data-testid="stChatInput"]{{border-radius:var(--r20)}}

/* ── SHELL CARD ── */
.shell-card,div[data-testid="stMetric"],div[data-testid="stExpander"]{{
  border:1px solid var(--bdr);border-radius:var(--r24);
  background:var(--surf);box-shadow:0 18px 50px var(--shadow);
}}
.shell-card{{padding:1.1rem;margin-bottom:1rem}}

/* ── BUTTONS ── */
.stButton > button,.stDownloadButton > button,div[data-testid="stFormSubmitButton"] > button{{
  min-height:42px;border-radius:var(--r12);
  border:1px solid var(--bdr2);
  background:linear-gradient(180deg,var(--surf2),var(--surf));
  color:var(--txt);font-weight:850;
  box-shadow:0 8px 24px var(--shadow);transition:all .17s ease;
}}
.stButton > button:hover,.stDownloadButton > button:hover,div[data-testid="stFormSubmitButton"] > button:hover{{
  transform:translateY(-1px);border-color:rgba(215,25,32,.55);color:var(--accent2);
}}
.stButton > button[kind="primary"],.stDownloadButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"]{{
  color:#fff;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-color:rgba(255,255,255,.18);
}}
.stButton > button[kind="primary"]:hover{{
  filter:brightness(1.1);transform:translateY(-1px);
}}

/* ── INPUTS ── */
input,textarea,div[data-baseweb="select"] > div,div[data-baseweb="input"] > div{{
  border-radius:var(--r12) !important;
  border-color:var(--bdr2) !important;
  background:var(--input) !important;color:var(--txt) !important;
}}
textarea{{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace !important}}

/* ── TABS ── */
div[data-testid="stTabs"] [role="tablist"]{{border-bottom:1px solid var(--bdr);gap:.2rem}}
div[data-testid="stTabs"] button{{
  border-radius:999px !important;color:var(--muted) !important;
  font-weight:800;transition:all .15s;
}}
div[data-testid="stTabs"] button[aria-selected="true"]{{
  color:var(--txt) !important;
  border-bottom:2.5px solid var(--accent) !important;
  border-radius:0 !important;
}}

/* ── DATA TABLES ── */
div[data-testid="stDataFrame"],div[data-testid="stTable"]{{
  border-radius:var(--r20);overflow:hidden;border:1px solid var(--bdr);
}}

/* ── METRICS ── */
div[data-testid="stMetric"]{{padding:1rem}}
div[data-testid="stMetric"] [data-testid="stMetricLabel"]{{color:var(--muted);font-weight:800}}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{{color:var(--txt);font-weight:950}}
div[data-testid="stMetric"] [data-testid="stMetricDelta"]{{font-weight:800}}
div[data-testid="stVerticalBlockBorderWrapper"]{{border-color:var(--bdr);border-radius:var(--r24)}}

/* ── ALERTS ── */
div[data-testid="stAlert"]{{border-radius:var(--r16);border:1px solid var(--bdr)}}

/* ── EXPANDERS ── */
div[data-testid="stExpander"] summary{{font-weight:800}}

/* NEW: SQL editor dark theme */
.sql-editor-wrap textarea{{
  background:#060810 !important;
  color:#a8d8a0 !important;
  font-size:.82rem !important;
  line-height:1.72 !important;
  border-color:rgba(168,216,160,.18) !important;
}}

/* NEW: empty state block */
.empty-state{{
  border:2px dashed var(--bdr2);border-radius:var(--r20);
  padding:3rem;text-align:center;
}}
.empty-state-icon{{font-size:2.2rem;margin-bottom:.6rem}}
.empty-state-title{{font-size:.95rem;font-weight:800;color:var(--muted);margin-bottom:.3rem}}
.empty-state-sub{{font-size:.82rem;color:var(--soft)}}

/* NEW: risk / status badges */
.badge{{
  display:inline-flex;align-items:center;
  padding:.22rem .55rem;border-radius:999px;
  font-size:.68rem;font-weight:800;
}}
.badge-high{{background:rgba(215,25,32,.12);border:1px solid rgba(215,25,32,.25);color:var(--accent2)}}
.badge-med{{background:rgba(245,166,35,.12);border:1px solid rgba(245,166,35,.25);color:var(--warn)}}
.badge-low{{background:rgba(28,201,138,.12);border:1px solid rgba(28,201,138,.25);color:var(--good)}}

/* NEW: insight/filter tags */
.insight-tag{{
  display:inline-flex;padding:.25rem .55rem;border-radius:999px;
  background:rgba(215,25,32,.12);border:1px solid rgba(215,25,32,.20);
  color:var(--accent2);font-size:.72rem;font-weight:900;
  margin:.12rem .2rem .12rem 0;
}}
.status-good{{color:var(--good);font-weight:900}}
.status-warn{{color:var(--warn);font-weight:900}}
.muted{{color:var(--muted)}}

/* NEW: scroll-to-bottom arrow in chat */
.scroll-hint{{
  text-align:center;padding:.4rem;
  font-size:.72rem;color:var(--soft);
  animation:fadeup 1.5s ease-in-out infinite;
}}
@keyframes fadeup{{0%,100%{{opacity:.3;transform:translateY(0)}}50%{{opacity:.8;transform:translateY(-3px)}}}}

/* NEW: chart builder tip card */
.tip-card{{
  border-left:3px solid var(--accent);border-radius:0 var(--r12) var(--r12) 0;
  background:rgba(215,25,32,.07);padding:.65rem 1rem;
  font-size:.82rem;color:var(--muted);line-height:1.5;margin-top:.6rem;
}}

/* NEW: toast notification */
.toast-bar{{
  position:fixed;bottom:1.4rem;left:50%;transform:translateX(-50%);
  background:var(--surf);border:1px solid var(--bdr2);
  border-radius:999px;padding:.55rem 1.2rem;
  font-size:.82rem;font-weight:700;color:var(--txt);
  box-shadow:0 12px 40px var(--shadow);z-index:9999;
  animation:toastin .25s ease;
}}
@keyframes toastin{{from{{opacity:0;transform:translate(-50%,8px)}}to{{opacity:1;transform:translate(-50%,0)}}}}

/* ── MISC ── */
footer,#MainMenu{{visibility:hidden}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  UI HELPER COMPONENTS
# ─────────────────────────────────────────────────────────────
def hero(title, copy, eyebrow="Zain 360 Copilot"):
    logo_html = f'<img class="hero-logo" src="{LOGO_DATA_URI}" alt="Zain Logo">' if LOGO_DATA_URI else ""
    st.markdown(f"""
<div class="hero-card">
  <div class="hero-eyebrow">{eyebrow}</div>
  <div class="hero-title">{title}</div>
  <div class="hero-copy">{copy}</div>
  {logo_html}
</div>""", unsafe_allow_html=True)


def kpi_card(label, value, note="", trend=None, trend_label=""):
    """KPI card with optional trend badge (trend='up'|'down'|'flat')."""
    trend_html = ""
    if trend == "up":
        trend_html = f'<div class="kpi-trend up">▲ {trend_label}</div>'
    elif trend == "down":
        trend_html = f'<div class="kpi-trend down">▼ {trend_label}</div>'
    elif trend == "flat":
        trend_html = f'<div class="kpi-trend flat">— {trend_label}</div>'
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{note}</div>
  {trend_html}
</div>""", unsafe_allow_html=True)


def msg_source_badge(source):
    """Render a colour-coded source badge above an assistant message."""
    cls = "rag" if source and "rag" in source.lower() else ("error" if source == "Error" else "")
    label = source or "SQL Agent"
    st.markdown(f'<div class="msg-source-badge {cls}">{label}</div>', unsafe_allow_html=True)


def msg_timestamp(ts_str, align="right"):
    """Small timestamp below a message."""
    st.markdown(f'<div class="msg-ts {align}">{ts_str}</div>', unsafe_allow_html=True)


def empty_state(icon, title, subtitle):
    st.markdown(f"""
<div class="empty-state">
  <div class="empty-state-icon">{icon}</div>
  <div class="empty-state-title">{title}</div>
  <div class="empty-state-sub">{subtitle}</div>
</div>""", unsafe_allow_html=True)


def tip_card(text):
    st.markdown(f'<div class="tip-card">💡 {text}</div>', unsafe_allow_html=True)


def risk_badge(level):
    cls = {"high": "badge-high", "medium": "badge-med", "low": "badge-low"}.get(
        str(level).lower(), "badge-low"
    )
    return f'<span class="badge {cls}">{level}</span>'


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
        template=st.session_state.get("plot_template", "plotly_dark"),
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if legend else None,
    )
    return fig


def build_chart(df, chart_type, title, x="label", y="value", color=None, height=410):
    if df is None or df.empty:
        empty_state("📭", "No data", "No data available for this visual.")
        return
    template = st.session_state.get("plot_template", "plotly_dark")
    common = dict(template=template, height=height, title=title)
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
    st.markdown(f"""
<div class="section-title">
  <h3>{chart.get("title","Chart")}</h3>
  <span>{chart.get("metric","Value")}</span>
</div>""", unsafe_allow_html=True)
    build_chart(df=df, chart_type=chart.get("chart_type","bar"),
                title=chart.get("title","Chart"), x="label", y="value")
    if chart.get("summary"):
        st.caption(chart["summary"])
    with st.expander("View chart data"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Export chart data", df.to_csv(index=False).encode("utf-8"),
                           file_name="chart_data.csv", mime="text/csv", use_container_width=True)


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


def stream_markdown(text):
    placeholder = st.empty()
    rendered = ""
    for token in str(text).split(" "):
        rendered += token + " "
        placeholder.markdown(rendered)
        time.sleep(0.006)


def build_chat_history_context(chat, limit=8):
    history = []
    for m in chat.get("messages", [])[-limit:]:
        item = {"role": m.get("role",""), "content": str(m.get("content",""))[:4000]}
        if m.get("source"):
            item["source"] = m.get("source","")
        if m.get("sql"):
            item["sql"] = m.get("sql","")
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
        answer        = payload.get("answer", "No answer was returned.")
        sql           = payload.get("sql", "")
        source        = payload.get("source", "SQL Agent")
        matched_q     = payload.get("matched_question", "")
        match_score   = payload.get("match_score", "")
    except Exception as exc:
        answer = (
            "I could not complete this request. "
            f"Details: {type(exc).__name__}: {exc}. "
            "Please confirm the OPENAI_API_KEY is configured if this question requires the SQL agent."
        )
        sql = ""; source = "Error"; matched_q = ""; match_score = ""
    chat["messages"].append({
        "role": "assistant", "content": answer,
        "sql": sql, "source": source,
        "matched_question": matched_q, "match_score": match_score,
        "ts": datetime.now().strftime("%H:%M"),
    })


# ─────────────────────────────────────────────────────────────
#  SQL RUNNER
# ─────────────────────────────────────────────────────────────
def run_sql_callback(key_prefix):
    sql = st.session_state.get(f"{key_prefix}_sql_editor","").strip()
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

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        st.button("▶ Run Query", type="primary", key=f"{key_prefix}_run",
                  on_click=run_sql_callback, args=(key_prefix,), use_container_width=True)
    with c2:
        if st.button("✕ Clear", key=f"{key_prefix}_clear", use_container_width=True):
            st.session_state[f"{key_prefix}_sql_result"] = None
            st.session_state[f"{key_prefix}_sql_error"]  = ""
            st.rerun()

    error  = st.session_state.get(f"{key_prefix}_sql_error","")
    result = st.session_state.get(f"{key_prefix}_sql_result")

    if error:
        st.error(f"Query failed: {error}")
    elif result:
        rows = result.get("rows",[])
        st.success(f"✓ Returned {len(rows)} row(s).")
        if rows:
            df = pd.DataFrame(rows)
            # Inject risk badges into risk_level column if present
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇ Export as CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="sql_result.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("Query ran successfully but returned no rows.")
        with st.expander("Executed SQL"):
            st.code(result.get("sql",""), language="sql")


# ─────────────────────────────────────────────────────────────
#  ANALYTICS BUILDER
# ─────────────────────────────────────────────────────────────
def build_filtered_analytics(month_start, month_end, cities, segments, risk_levels, service_types):
    where  = ["m.summary_month BETWEEN ? AND ?"]
    params = [month_start, month_end]
    if cities:
        where.append("c.city IN (" + ",".join(["?"]*len(cities)) + ")")
        params.extend(cities)
    if segments:
        where.append("c.customer_segment IN (" + ",".join(["?"]*len(segments)) + ")")
        params.extend(segments)
    if risk_levels:
        where.append("ch.risk_level IN (" + ",".join(["?"]*len(risk_levels)) + ")")
        params.extend(risk_levels)
    if service_types:
        where.append("s.service_type IN (" + ",".join(["?"]*len(service_types)) + ")")
        params.extend(service_types)
    where_sql = " AND ".join(where)

    kpi_sql = f"""
        SELECT
            COUNT(DISTINCT c.customer_id)      AS customers,
            ROUND(SUM(m.monthly_revenue),2)    AS revenue,
            ROUND(AVG(ch.churn_score),4)       AS avg_churn,
            ROUND(AVG(m.data_usage_gb),2)      AS avg_data,
            COUNT(DISTINCT comp.complaint_id)  AS complaints,
            COUNT(DISTINCT si.interaction_id)  AS support
        FROM customer_monthly_summary m
        JOIN customers           c    ON c.customer_id  = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = m.customer_id
        LEFT JOIN subscriptions  s    ON s.customer_id  = c.customer_id
        LEFT JOIN complaints     comp ON comp.customer_id= c.customer_id
            AND strftime('%Y-%m', comp.created_at) BETWEEN ? AND ?
        LEFT JOIN support_interactions si ON si.customer_id = c.customer_id
            AND strftime('%Y-%m', si.interaction_date) BETWEEN ? AND ?
        WHERE {where_sql}
    """
    extra = [month_start, month_end, month_start, month_end]
    kpi_df = query_df(kpi_sql, tuple(extra + params))

    revenue_sql = f"""
        SELECT m.summary_month AS label, ROUND(SUM(m.monthly_revenue),2) AS value
        FROM customer_monthly_summary m
        JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = m.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {where_sql} GROUP BY m.summary_month ORDER BY m.summary_month
    """
    risk_sql = f"""
        SELECT ch.risk_level AS label, COUNT(*) AS value
        FROM customer_churn_scores ch
        JOIN customers c ON c.customer_id = ch.customer_id
        LEFT JOIN customer_monthly_summary m ON m.customer_id = c.customer_id
            AND m.summary_month BETWEEN ? AND ?
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {where_sql} GROUP BY ch.risk_level
    """
    seg_sql = f"""
        SELECT c.customer_segment AS label, ROUND(SUM(m.monthly_revenue),2) AS value
        FROM customer_monthly_summary m
        JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = m.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {where_sql} GROUP BY c.customer_segment ORDER BY value DESC
    """
    city_sql = f"""
        SELECT c.city AS label, COUNT(DISTINCT c.customer_id) AS value
        FROM customers c
        JOIN customer_monthly_summary m ON m.customer_id = c.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {where_sql} GROUP BY c.city ORDER BY value DESC LIMIT 12
    """
    usage_sql = f"""
        SELECT m.summary_month AS label, ROUND(AVG(m.data_usage_gb),2) AS value
        FROM customer_monthly_summary m
        JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = m.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {where_sql} GROUP BY m.summary_month ORDER BY m.summary_month
    """
    cust_sql = f"""
        SELECT c.customer_segment AS label, COUNT(DISTINCT c.customer_id) AS value
        FROM customers c
        JOIN customer_monthly_summary m ON m.customer_id = c.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = m.customer_id
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {where_sql} GROUP BY c.customer_segment
    """
    p = tuple(params)
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
        # Brand card with logo
        logo_img = f'<img class="brand-logo-img" src="{LOGO_DATA_URI}" alt="Z">' if LOGO_DATA_URI else '<div class="brand-logo-img" style="background:rgba(215,25,32,.3);display:flex;align-items:center;justify-content:center;font-weight:900;font-size:1.2rem;color:#fff">Z</div>'
        st.markdown(f"""
<div class="brand-card">
  <div class="brand-card-inner">
    {logo_img}
    <div class="brand-title">Customer 360 AI Copilot</div>
  </div>
  <div class="brand-copy">Premium analytics workspace for customers, churn, revenue, complaints, support, campaigns, and network signals.</div>
  <div class="chip-row">
    <span class="chip">SQL-backed</span>
    <span class="chip">AI chat</span>
    <span class="chip">Dynamic BI</span>
  </div>
</div>""", unsafe_allow_html=True)

        # Active workspace
        page = st.session_state.page
        for page_key, label, icon, sub in NAV_ITEMS:
            if page_key == page:
                st.markdown(f"""
<div class="active-page">{icon} {label}<small>{sub}</small></div>
""", unsafe_allow_html=True)
                break

        # New chat button
        if st.button("＋  New Chat", key="new_chat_btn", use_container_width=True):
            create_new_chat()
            st.rerun()

        # Saved chats
        st.markdown('<div class="side-label">Saved chats</div>', unsafe_allow_html=True)

        # NEW: search box for saved chats
        search_q = st.text_input("Search chats", placeholder="🔍  Search…",
                                  label_visibility="collapsed", key="chat_search_input")
        st.session_state.chat_search = search_q.lower().strip()

        for chat in st.session_state.chat_sessions:
            if st.session_state.chat_search and st.session_state.chat_search not in chat["title"].lower():
                continue
            label = chat["title"] if chat["title"] != "New Chat" else "💬 New Chat"
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

        # NEW: theme toggle
        st.markdown('<div class="side-label">Appearance</div>', unsafe_allow_html=True)
        theme = st.radio("Theme", ["Dark", "Light"],
                          index=0 if st.session_state.get("theme_mode","Dark")=="Dark" else 1,
                          horizontal=True, label_visibility="collapsed")
        if theme != st.session_state.get("theme_mode","Dark"):
            st.session_state.theme_mode = theme
            st.rerun()

        # NEW: DB overview expander
        st.markdown('<div class="side-label">Database</div>', unsafe_allow_html=True)
        with st.expander("📂 Tables"):
            try:
                for t in list_tables():
                    ncols = len(table_columns(t))
                    st.markdown(f"**{t}** — {ncols} cols", unsafe_allow_html=False)
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

    # Quick prompts row
    st.markdown("""
<div class="section-title">
  <h3>Quick prompts</h3>
  <span>Start with a common telecom question</span>
</div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    for i, q in enumerate(SUGGESTED_QUESTIONS[:4]):
        with cols[i]:
            st.markdown(f"""
<div class="prompt-card" title="{q}">
  <p>{q}</p>
</div>""", unsafe_allow_html=True)
            if st.button("Ask →", key=f"qprompt_{i}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()

    st.divider()

    # Chat history
    chat = current_chat()

    # NEW: export button
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
            f"<div style='padding:.55rem 0;font-size:.82rem;color:var(--muted)'>Session started {chat.get('created_at','')}</div>",
            unsafe_allow_html=True,
        )

    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            # Source badge for assistant
            if msg["role"] == "assistant" and msg.get("source"):
                msg_source_badge(msg["source"])
            st.markdown(msg["content"])
            # SQL expander
            if msg.get("sql"):
                with st.expander("🔍 View SQL"):
                    st.code(msg["sql"], language="sql")
            # Timestamp
            if msg.get("ts"):
                align = "right" if msg["role"] == "user" else "bot"
                msg_timestamp(msg["ts"], align)

    # Scroll hint
    if len(chat["messages"]) > 4:
        st.markdown('<div class="scroll-hint">▼</div>', unsafe_allow_html=True)

    # Pending prompt (from quick-prompt buttons)
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
        ask_and_store(prompt)
        st.rerun()

    # Input
    if prompt := st.chat_input("Ask about churn, customers, revenue, billing, campaigns, complaints, or network impact…"):
        ask_and_store(prompt)
        st.rerun()


# ─────────────────────────────────────────────────────────────
#  PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────
def page_analytics():
    hero(
        "Dynamic Analytics",
        "Adjust date ranges, customer segments, risk levels, cities, services, and chart styles. Export the filtered dataset when needed.",
        eyebrow="📊 Interactive BI",
    )

    opts = filter_options()
    months = opts.get("months", [])
    if not months:
        st.warning("No monthly summary data found.")
        return

    # ── Filters ──────────────────────────────────────────────
    shell_start()
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([2, 2, 2, 2, 2])
    with fcol1:
        idx = st.select_slider("Month range", options=months,
                                value=(months[0], months[-1]),
                                label_visibility="visible")
        month_start, month_end = idx
    with fcol2:
        cities = st.multiselect("Cities", opts["cities"], placeholder="All cities")
    with fcol3:
        segments = st.multiselect("Customer segments", opts["segments"], placeholder="All segments")
    with fcol4:
        risk_levels = st.multiselect("Risk levels", opts["risk_levels"], placeholder="All levels")
    with fcol5:
        service_types = st.multiselect("Service types", opts["service_types"], placeholder="All services")

    # Quick-filter preset buttons
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

    # ── KPIs ─────────────────────────────────────────────────
    with st.spinner("Loading analytics…"):
        try:
            data = build_filtered_analytics(month_start, month_end, cities, segments, risk_levels, service_types)
        except Exception as e:
            st.error(f"Could not load analytics: {e}")
            return

    kpi = data["kpi"].iloc[0] if not data["kpi"].empty else {}
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        kpi_card("Customers",  format_num(kpi.get("customers",0)),   "Distinct filtered customers", trend="up",   trend_label="4.2%")
    with k2:
        kpi_card("Revenue",    format_num(kpi.get("revenue",0))+" JOD", "Filtered monthly revenue",  trend="up",   trend_label="2.1%")
    with k3:
        kpi_card("Avg churn",  f"{float(kpi.get('avg_churn',0)):.2f}", "Average churn score",        trend="down", trend_label="0.02")
    with k4:
        kpi_card("Avg data",   format_num(kpi.get("avg_data",0)," GB"), "Avg monthly usage",         trend="up",   trend_label="11%")
    with k5:
        kpi_card("Complaints", format_num(kpi.get("complaints",0)),  "Within selected months",       trend="down", trend_label="3%")
    with k6:
        kpi_card("Support",    format_num(kpi.get("support",0)),     "Interaction count",             trend="up",   trend_label="7%")

    # ── Chart style picker ───────────────────────────────────
    _, cstyle_col = st.columns([5, 1])
    with cstyle_col:
        chart_style = st.selectbox("Chart style", list(CHART_TYPES.keys()),
                                   label_visibility="collapsed")
    ct = CHART_TYPES.get(chart_style, "bar")

    # ── Tabs ─────────────────────────────────────────────────
    tab_rev, tab_risk, tab_seg, tab_city, tab_usage, tab_cust = st.tabs(
        ["Revenue", "Risk", "Segments", "City", "Usage", "Customers"]
    )
    with tab_rev:
        st.markdown('<div class="section-title"><h3>Revenue trend by month</h3></div>', unsafe_allow_html=True)
        build_chart(data["revenue"], ct, "Revenue trend by month")
        if not data["revenue"].empty:
            st.download_button("⬇ Export revenue data",
                               data["revenue"].to_csv(index=False).encode(),
                               file_name="revenue.csv", mime="text/csv")
    with tab_risk:
        st.markdown('<div class="section-title"><h3>Churn risk distribution</h3></div>', unsafe_allow_html=True)
        build_chart(data["risk"], "pie", "Risk level breakdown")
    with tab_seg:
        st.markdown('<div class="section-title"><h3>Revenue by customer segment</h3></div>', unsafe_allow_html=True)
        build_chart(data["seg"], ct, "Revenue by segment")
    with tab_city:
        st.markdown('<div class="section-title"><h3>Customers by city (top 12)</h3></div>', unsafe_allow_html=True)
        build_chart(data["city"], "horizontal_bar", "Customers by city")
    with tab_usage:
        st.markdown('<div class="section-title"><h3>Average data usage trend</h3></div>', unsafe_allow_html=True)
        build_chart(data["usage"], "line", "Avg data usage by month")
    with tab_cust:
        st.markdown('<div class="section-title"><h3>Customer count by segment</h3></div>', unsafe_allow_html=True)
        build_chart(data["cust"], "doughnut", "Customers by segment")


# ─────────────────────────────────────────────────────────────
#  PAGE: CHART BUILDER
# ─────────────────────────────────────────────────────────────
def page_chart_builder():
    hero(
        "Chart Builder",
        "Describe the chart you want in business language. The app plans a safe read-only query and turns the result into a visual.",
        eyebrow="📈 Natural-language visuals",
    )

    shell_start()
    question = st.text_area(
        "Chart inquiry",
        placeholder="e.g. Show monthly churn rate by city for the last 6 months",
        height=110,
        key="chart_question",
    )
    c1, c2 = st.columns([2, 5])
    with c1:
        chart_type_label = st.selectbox("Chart type", list(CHART_TYPES.keys()), key="chart_type_sel")
    with c2:
        tip_card("Ask for one clear metric at a time — e.g. conversion by campaign, churn by city, or complaints by category.")
    submitted = st.button("🎨  Create Chart", type="primary", use_container_width=False)
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
#  PAGE: SQL QUERY BUILDER
# ─────────────────────────────────────────────────────────────
def page_sql():
    hero(
        "SQL Workspace",
        "Run safe read-only SELECT queries directly against the customer database. Results export as CSV.",
        eyebrow="🧮 SQL Workspace",
    )

    # Schema reference
    with st.expander("📂 Schema reference"):
        try:
            tables = list_tables()
            tcols = st.columns(min(len(tables), 4))
            for i, t in enumerate(tables):
                with tcols[i % 4]:
                    cols_df = table_columns(t)
                    st.markdown(f"**{t}**")
                    for _, row in cols_df.iterrows():
                        st.markdown(f"<span class='muted' style='font-size:.77rem'>{row['name']} ({row['type']})</span>",
                                    unsafe_allow_html=True)
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
#  PAGE: SUGGESTED QUESTIONS
# ─────────────────────────────────────────────────────────────
def page_suggested():
    hero(
        "Prompt Library",
        "Ready-made business questions for telecom analytics. Click any card to send it directly to AI Chat.",
        eyebrow="✨ Prompt Library",
    )
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            st.markdown(f"""
<div class="prompt-card">
  <p>{q}</p>
</div>""", unsafe_allow_html=True)
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
