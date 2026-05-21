import base64
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
    """Load OPENAI_API_KEY from Streamlit secrets when deployed."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if key:
        os.environ["OPENAI_API_KEY"] = str(key)


load_streamlit_secret()

from class3_sql_agent_backend import (  # noqa: E402
    ask_sql_agent_payload,
    build_chart_from_question,
    execute_sql_query,
    get_database_overview,
)


st.set_page_config(
    page_title="Zain Customer 360 Copilot",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "⚡",
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
    ("Chat", "AI Chat", "💬", "Ask business questions"),
    ("Analytics", "Dynamic Analytics", "📊", "Filter KPIs and charts"),
    ("Customer Insights", "Customer Insights", "👤", "Inspect one customer"),
    ("Chart Builder", "Chart Builder", "📈", "Create custom visuals"),
    ("SQL Query Builder", "SQL Workspace", "🧮", "Run safe SELECT queries"),
    ("Suggested Questions", "Prompt Library", "✨", "Ready-made use cases"),
    ("Data Catalog", "Data Catalog", "🗂️", "Browse tables and fields"),
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


def inject_css():
    dark = st.session_state.get("theme_mode", "Dark") == "Dark"
    if dark:
        palette = {
            "bg": "#080A0F",
            "bg2": "#0E1118",
            "surface": "rgba(17, 21, 32, 0.88)",
            "surface2": "rgba(25, 31, 45, 0.82)",
            "border": "rgba(255,255,255,.10)",
            "border2": "rgba(255,255,255,.16)",
            "text": "#F7F8FA",
            "muted": "#AAB2C0",
            "soft": "#7F8999",
            "accent": "#D71920",
            "accent2": "#FF4D57",
            "good": "#31D0AA",
            "warn": "#F6C85F",
            "shadow": "rgba(0,0,0,.38)",
            "input": "rgba(255,255,255,.055)",
            "plot_template": "plotly_dark",
        }
    else:
        palette = {
            "bg": "#F5F7FB",
            "bg2": "#FFFFFF",
            "surface": "rgba(255, 255, 255, 0.92)",
            "surface2": "rgba(247, 249, 253, 0.95)",
            "border": "rgba(13, 18, 30, .10)",
            "border2": "rgba(13, 18, 30, .16)",
            "text": "#141821",
            "muted": "#566173",
            "soft": "#7A8494",
            "accent": "#D71920",
            "accent2": "#B91018",
            "good": "#058A6E",
            "warn": "#A96F00",
            "shadow": "rgba(23,31,56,.12)",
            "input": "rgba(255,255,255,.90)",
            "plot_template": "plotly_white",
        }
    st.session_state.plot_template = palette["plot_template"]

    st.markdown(
        f"""
        <style>
          :root {{
            --app-bg: {palette['bg']};
            --app-bg-2: {palette['bg2']};
            --surface: {palette['surface']};
            --surface-2: {palette['surface2']};
            --border: {palette['border']};
            --border-2: {palette['border2']};
            --text: {palette['text']};
            --muted: {palette['muted']};
            --soft: {palette['soft']};
            --accent: {palette['accent']};
            --accent-2: {palette['accent2']};
            --good: {palette['good']};
            --warn: {palette['warn']};
            --shadow: {palette['shadow']};
            --input: {palette['input']};
            --radius-lg: 24px;
            --radius-md: 18px;
            --radius-sm: 12px;
          }}

          .stApp {{
            background:
              radial-gradient(circle at 0% -10%, rgba(215, 25, 32, .22), transparent 32%),
              radial-gradient(circle at 95% 0%, rgba(135, 25, 32, .12), transparent 24%),
              linear-gradient(135deg, var(--app-bg), var(--app-bg-2));
            color: var(--text);
          }}

          .block-container {{
            padding: 1.35rem 2rem 4.25rem;
            max-width: 1440px;
          }}

          @media (max-width: 900px) {{
            .block-container {{
              padding: 1rem 1rem 5rem;
            }}
          }}

          h1, h2, h3, h4, h5, h6, p, label, span {{
            color: var(--text);
          }}

          h1 {{
            letter-spacing: -0.045em;
            font-weight: 900;
            margin-bottom: .2rem;
          }}

          h2, h3 {{
            letter-spacing: -0.025em;
          }}

          a {{
            color: var(--accent-2);
          }}

          section[data-testid="stSidebar"] {{
            background:
              radial-gradient(circle at 15% 5%, rgba(215,25,32,.25), transparent 30%),
              linear-gradient(180deg, rgba(15,18,27,.97), rgba(9,11,17,.98)) !important;
            border-right: 1px solid rgba(255,255,255,.11);
          }}

          section[data-testid="stSidebar"] * {{
            color: #F7F8FA;
          }}

          section[data-testid="stSidebar"] > div {{
            padding: 1rem .9rem 1.4rem;
          }}

          [data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            min-height: 44px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,.10);
            background: rgba(255,255,255,.055);
            color: #F7F8FA;
            font-weight: 750;
            justify-content: flex-start;
            text-align: left;
            padding: .68rem .85rem;
            box-shadow: none;
            transition: all .18s ease;
          }}

          [data-testid="stSidebar"] .stButton > button:hover {{
            transform: translateY(-1px);
            border-color: rgba(255,255,255,.24);
            background: rgba(255,255,255,.10);
          }}

          .brand-card {{
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 24px;
            padding: 1rem;
            background:
              linear-gradient(135deg, rgba(215,25,32,.24), rgba(255,255,255,.055)),
              rgba(255,255,255,.06);
            box-shadow: 0 20px 60px rgba(0,0,0,.25);
            margin: .25rem 0 1rem;
          }}

          .brand-title {{
            font-size: 1.02rem;
            font-weight: 900;
            letter-spacing: -.02em;
            margin-bottom: .35rem;
          }}

          .brand-copy {{
            font-size: .77rem;
            line-height: 1.5;
            color: rgba(247,248,250,.72);
          }}

          .chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: .38rem;
            margin-top: .85rem;
          }}

          .chip {{
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 999px;
            padding: .26rem .52rem;
            font-size: .67rem;
            font-weight: 850;
            color: #fff;
            background: rgba(255,255,255,.075);
          }}

          .side-label {{
            margin: 1rem .15rem .45rem;
            color: rgba(247,248,250,.54);
            font-size: .68rem;
            text-transform: uppercase;
            letter-spacing: .12em;
            font-weight: 900;
          }}

          .active-page {{
            border: 1px solid rgba(215,25,32,.55);
            background:
              linear-gradient(135deg, rgba(215,25,32,.25), rgba(255,255,255,.075));
            border-radius: 16px;
            padding: .72rem .85rem;
            margin: .25rem 0 .55rem;
            color: #fff;
            box-shadow: 0 0 0 4px rgba(215,25,32,.12);
            font-weight: 900;
          }}

          .active-page small {{
            display: block;
            color: rgba(247,248,250,.68);
            font-size: .68rem;
            font-weight: 600;
            margin-top: .12rem;
          }}

          .shell-card, div[data-testid="stMetric"], div[data-testid="stExpander"] {{
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            background: var(--surface);
            box-shadow: 0 18px 50px var(--shadow);
          }}

          .shell-card {{
            padding: 1.05rem;
            margin-bottom: 1rem;
          }}

          .hero-card {{
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1.3rem min(11.5rem, 18vw) 1.3rem 1.35rem;
            background:
              radial-gradient(circle at 10% 0%, rgba(215,25,32,.24), transparent 28%),
              linear-gradient(135deg, var(--surface), var(--surface-2));
            box-shadow: 0 22px 70px var(--shadow);
            margin-bottom: 1.1rem;
          }}

          .hero-card::after {{
            content: "";
            position: absolute;
            right: 1.35rem;
            top: 50%;
            width: clamp(76px, 10vw, 132px);
            height: clamp(76px, 10vw, 132px);
            transform: translateY(-50%);
            border-radius: 28px;
            background: rgba(255,255,255,.045);
            border: 1px solid var(--border);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.10);
          }}

          .hero-logo {{
            position: absolute;
            right: 2.15rem;
            top: 50%;
            width: clamp(58px, 7.4vw, 96px);
            transform: translateY(-50%);
            opacity: .64;
            filter: grayscale(.1);
            z-index: 1;
            pointer-events: none;
          }}

          .hero-eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .32rem .62rem;
            border: 1px solid var(--border-2);
            border-radius: 999px;
            background: rgba(215,25,32,.10);
            color: var(--accent-2);
            font-size: .72rem;
            font-weight: 900;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .8rem;
          }}

          .hero-title {{
            font-size: clamp(1.8rem, 3vw, 3.15rem);
            font-weight: 950;
            letter-spacing: -.06em;
            line-height: 1.02;
            color: var(--text);
            max-width: 900px;
          }}

          .hero-copy {{
            color: var(--muted);
            font-size: .98rem;
            line-height: 1.6;
            max-width: 880px;
            margin-top: .65rem;
          }}

          @media (max-width: 700px) {{
            .hero-card {{
              padding: 1.15rem;
            }}

            .hero-card::after,
            .hero-logo {{
              display: none;
            }}
          }}

          .kpi-card {{
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1rem;
            min-height: 128px;
            background:
              linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02)),
              var(--surface);
            box-shadow: 0 18px 46px var(--shadow);
          }}

          .kpi-label {{
            color: var(--muted);
            font-size: .78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
          }}

          .kpi-value {{
            color: var(--text);
            font-size: clamp(1.45rem, 2.6vw, 2.05rem);
            font-weight: 950;
            line-height: 1;
            margin: .58rem 0 .35rem;
            letter-spacing: -.04em;
          }}

          .kpi-note {{
            color: var(--soft);
            font-size: .78rem;
            line-height: 1.4;
          }}

          .section-title {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: .7rem 0 .7rem;
          }}

          .section-title h3 {{
            margin: 0;
            font-size: 1.15rem;
            font-weight: 900;
          }}

          .section-title span {{
            color: var(--muted);
            font-size: .82rem;
          }}

          .prompt-card {{
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: .9rem;
            background: var(--surface);
            box-shadow: 0 14px 36px var(--shadow);
            min-height: 132px;
          }}

          .prompt-card b {{
            display: block;
            margin-bottom: .35rem;
          }}

          .prompt-card p {{
            margin: 0;
            color: var(--muted);
            font-size: .85rem;
            line-height: 1.45;
          }}

          .insight-tag {{
            display: inline-flex;
            padding: .25rem .55rem;
            border-radius: 999px;
            background: rgba(215,25,32,.12);
            border: 1px solid rgba(215,25,32,.20);
            color: var(--accent-2);
            font-size: .72rem;
            font-weight: 900;
            margin: .12rem .2rem .12rem 0;
          }}

          .status-good {{
            color: var(--good);
            font-weight: 900;
          }}

          .status-warn {{
            color: var(--warn);
            font-weight: 900;
          }}

          .muted {{
            color: var(--muted);
          }}

          div[data-testid="stMetric"] {{
            padding: 1rem;
          }}

          div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
            color: var(--muted);
            font-weight: 800;
          }}

          div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: var(--text);
            font-weight: 950;
          }}

          div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: var(--border);
            border-radius: var(--radius-lg);
          }}

          div[data-testid="stTabs"] button {{
            border-radius: 999px !important;
            color: var(--muted) !important;
            font-weight: 800;
          }}

          div[data-testid="stTabs"] button[aria-selected="true"] {{
            color: var(--text) !important;
          }}

          .stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {{
            min-height: 42px;
            border-radius: 14px;
            border: 1px solid var(--border-2);
            background: linear-gradient(180deg, var(--surface-2), var(--surface));
            color: var(--text);
            font-weight: 850;
            box-shadow: 0 10px 28px var(--shadow);
            transition: all .18s ease;
          }}

          .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{
            transform: translateY(-1px);
            border-color: rgba(215,25,32,.55);
            color: var(--accent-2);
          }}

          .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] {{
            color: #fff;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            border-color: rgba(255,255,255,.18);
          }}

          input, textarea, div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
            border-radius: 14px !important;
            border-color: var(--border-2) !important;
            background: var(--input) !important;
            color: var(--text) !important;
          }}

          textarea {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
          }}

          div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid var(--border);
          }}

          div[data-testid="stAlert"] {{
            border-radius: 16px;
            border: 1px solid var(--border);
          }}

          .stChatMessage {{
            border-radius: 22px;
            border: 1px solid var(--border);
            background: var(--surface);
            box-shadow: 0 12px 36px var(--shadow);
            padding: .75rem;
          }}

          [data-testid="stChatInput"] {{
            border-radius: 20px;
          }}

          footer, #MainMenu {{
            visibility: hidden;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, copy, eyebrow="Zain 360 Copilot"):
    logo_html = f'<img class="hero-logo" src="{LOGO_DATA_URI}" alt="Zain Logo">' if LOGO_DATA_URI else ""
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-eyebrow">{eyebrow}</div>
          <div class="hero-title">{title}</div>
          <div class="hero-copy">{copy}</div>
          {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        if value.is_integer():
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
        st.info("No data available for this visual.")
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
    st.markdown(
        f"""
        <div class="section-title">
          <h3>{chart.get("title", "Chart")}</h3>
          <span>{chart.get("metric", "Value")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    build_chart(
        df=df,
        chart_type=chart.get("chart_type", "bar"),
        title=chart.get("title", "Chart"),
        x="label",
        y="value",
    )
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


def chat_to_markdown(chat):
    lines = [f"# {chat['title']}", f"Created: {chat.get('created_at', '')}", ""]
    for message in chat["messages"]:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"## {role}")
        if message.get("source"):
            lines.append(f"Agent: {message['source']}")
            lines.append("")
        lines.append(message.get("content", ""))
        if message.get("sql"):
            lines.append("")
            lines.append("```sql")
            lines.append(message["sql"])
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def stream_markdown(text):
    placeholder = st.empty()
    rendered = ""
    for token in str(text).split(" "):
        rendered += token + " "
        placeholder.markdown(rendered)
        time.sleep(0.006)


def ask_and_store(prompt):
    prompt = str(prompt).strip()
    if not prompt:
        return
    chat = current_chat()
    if chat["title"] == "New Chat":
        chat["title"] = title_from_question(prompt)
    chat["messages"].append({"role": "user", "content": prompt, "sql": ""})

    try:
        with st.spinner("Analyzing the database and preparing the answer..."):
            payload = ask_sql_agent_payload(prompt)
        answer = payload.get("answer", "No answer was returned.")
        sql = payload.get("sql", "")
        source = payload.get("source", "SQL Agent")
        matched_question = payload.get("matched_question", "")
        match_score = payload.get("match_score", "")
    except Exception as exc:
        answer = (
            "I could not complete this request. "
            f"Details: {type(exc).__name__}: {exc}. "
            "Please confirm the OPENAI_API_KEY is configured if this question requires the SQL agent."
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

    st.text_area(
        "SQL",
        height=190,
        key=editor_key,
        help="Only safe read-only SELECT queries are allowed.",
    )
    cols = st.columns([1, 1, 3])
    with cols[0]:
        st.button(
            "Run Query",
            type="primary",
            key=f"{key_prefix}_run_button",
            on_click=run_sql_callback,
            args=(key_prefix,),
            use_container_width=True,
        )
    with cols[1]:
        if st.button("Clear Result", key=f"{key_prefix}_clear", use_container_width=True):
            st.session_state[f"{key_prefix}_sql_result"] = None
            st.session_state[f"{key_prefix}_sql_error"] = ""
            st.rerun()

    error = st.session_state.get(f"{key_prefix}_sql_error", "")
    result = st.session_state.get(f"{key_prefix}_sql_result")
    if error:
        st.error(f"Query failed: {error}")
    elif result:
        rows = result.get("rows", [])
        st.success(f"Returned {len(rows)} row(s).")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Export result as CSV",
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
            m.summary_month,
            c.customer_id,
            c.full_name,
            c.city,
            c.governorate,
            c.customer_segment,
            c.customer_type,
            c.status AS customer_status,
            s.service_type,
            ch.churn_score,
            ch.risk_level,
            ch.main_risk_reason,
            ch.recommended_action,
            vs.value_segment,
            vs.arpu_jod,
            vs.total_revenue_6m_jod,
            m.total_revenue_jod,
            m.voice_minutes,
            m.data_used_gb,
            m.sms_count,
            m.support_interactions_count,
            m.complaints_count,
            m.payment_delay_days
        FROM customer_monthly_summary m
        JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments vs ON vs.customer_id = c.customer_id
        LEFT JOIN subscriptions s ON s.subscription_id = m.subscription_id
        WHERE {where_sql}
    """
    df = query_df(sql, tuple(params))
    return df, sql, params


def show_chat():
    chat = current_chat()
    hero(
        "Customer 360 Chat",
        "Ask direct business questions and inspect the SQL behind answers. Your chat sessions are saved during this browser session.",
        "Conversational analytics",
    )

    st.markdown('<div class="section-title"><h3>Quick prompts</h3><span>Start with a common telecom question</span></div>', unsafe_allow_html=True)
    prompt_cols = st.columns(4)
    quick_prompts = SUGGESTED_QUESTIONS[:4]
    for index, question in enumerate(quick_prompts):
        with prompt_cols[index]:
            if st.button(question, key=f"quick_{index}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()

    st.divider()

    for index, message in enumerate(chat["messages"]):
        with st.chat_message(message["role"]):
            if message.get("source"):
                st.caption(f"Answered by: {message['source']}")
            st.markdown(message.get("content", ""))
            if message.get("source") == "RAG Agent" and message.get("matched_question"):
                score = message.get("match_score", "")
                score_text = f" Similarity: {score}" if score != "" else ""
                st.caption(f"Memory match: {message['matched_question']}.{score_text}")
            if message.get("sql"):
                with st.expander("SQL visibility"):
                    render_sql_runner(message["sql"], key_prefix=f"{chat['id']}_history_{index}")

    if st.session_state.pending_prompt:
        pending = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
        ask_and_store(pending)
        st.rerun()

    prompt = st.chat_input("Ask about churn, customers, revenue, billing, campaigns, complaints, or network impact...")
    if prompt:
        ask_and_store(prompt)
        st.rerun()


def show_overview():
    hero(
        "Executive Overview",
        "A polished command-center view of database coverage, customer distribution, churn risk, network impact, and campaign activity.",
        "Database health",
    )
    data = get_database_overview()

    totals = {
        "customers": query_df("SELECT COUNT(*) AS v FROM customers")["v"].iloc[0],
        "revenue": query_df("SELECT SUM(total_revenue_jod) AS v FROM customer_monthly_summary")["v"].iloc[0],
        "open_complaints": query_df("SELECT COUNT(*) AS v FROM complaints WHERE status != 'Resolved'")["v"].iloc[0],
        "high_risk": query_df("SELECT COUNT(*) AS v FROM customer_churn_scores WHERE risk_level = 'High'")["v"].iloc[0],
        "overdue": query_df("SELECT COUNT(*) AS v FROM invoices WHERE days_overdue > 0")["v"].iloc[0],
        "network": query_df("SELECT SUM(affected_customers) AS v FROM network_events")["v"].iloc[0],
    }

    cols = st.columns(6)
    with cols[0]:
        kpi_card("Customers", format_num(totals["customers"]), "Total customer base")
    with cols[1]:
        kpi_card("Revenue", format_num(totals["revenue"], " JOD"), "Monthly summary total")
    with cols[2]:
        kpi_card("Open Complaints", format_num(totals["open_complaints"]), "Needs follow-up")
    with cols[3]:
        kpi_card("High Risk", format_num(totals["high_risk"]), "Churn priority")
    with cols[4]:
        kpi_card("Overdue Invoices", format_num(totals["overdue"]), "Billing attention")
    with cols[5]:
        kpi_card("Network Impact", format_num(totals["network"]), "Affected customers")

    tab1, tab2, tab3 = st.tabs(["Overview charts", "Operational tables", "Recommended actions"])

    with tab1:
        chart_cols = st.columns(2)
        for index, chart in enumerate(data["charts"]):
            with chart_cols[index % 2]:
                shell_start()
                render_chart(chart)
                shell_end()

    with tab2:
        table_df = pd.DataFrame(data["tables"])
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Export table inventory",
            table_df.to_csv(index=False).encode("utf-8"),
            file_name="database_tables.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab3:
        action_cols = st.columns(3)
        cards = [
            ("Retention", "Prioritize customers with high churn score, recent complaints, and overdue invoices."),
            ("Network", "Review cities with repeated affected-customer events and open critical network tickets."),
            ("Campaigns", "Compare conversion rate by campaign and focus spend on the highest-yield segment/channel pairs."),
        ]
        for col, (title, copy) in zip(action_cols, cards):
            with col:
                st.markdown(
                    f"""
                    <div class="prompt-card">
                      <b>{title}</b>
                      <p>{copy}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def show_dynamic_analytics():
    hero(
        "Dynamic Analytics",
        "Adjust date ranges, customer segments, risk levels, cities, services, and chart styles. Export the filtered dataset when needed.",
        "Interactive BI",
    )
    options = filter_options()
    months = options["months"]
    if not months:
        st.error("No monthly summary data is available.")
        return

    with st.container(border=True):
        filter_cols = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1])
        with filter_cols[0]:
            month_start, month_end = st.select_slider(
                "Month range",
                options=months,
                value=(months[0], months[-1]),
            )
        with filter_cols[1]:
            cities = st.multiselect("Cities", options["cities"], default=[])
        with filter_cols[2]:
            segments = st.multiselect("Customer segments", options["segments"], default=[])
        with filter_cols[3]:
            risk_levels = st.multiselect("Risk levels", options["risk_levels"], default=[])
        with filter_cols[4]:
            service_types = st.multiselect("Service types", options["service_types"], default=[])
        with filter_cols[5]:
            chart_label = st.selectbox("Chart style", list(CHART_TYPES.keys()), index=0)

    quick_cols = st.columns(4)
    with quick_cols[0]:
        if st.button("High-risk only", use_container_width=True):
            risk_levels = ["High"]
    with quick_cols[1]:
        if st.button("VIP customers", use_container_width=True):
            segments = ["VIP"] if "VIP" in options["segments"] else segments
    with quick_cols[2]:
        if st.button("Amman view", use_container_width=True):
            cities = ["Amman"] if "Amman" in options["cities"] else cities
    with quick_cols[3]:
        if st.button("All filters reset", use_container_width=True):
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

    kcols = st.columns(6)
    with kcols[0]:
        kpi_card("Customers", format_num(unique_customers), "Distinct filtered customers")
    with kcols[1]:
        kpi_card("Revenue", format_num(total_revenue, " JOD"), "Filtered monthly revenue")
    with kcols[2]:
        kpi_card("Avg Churn", f"{avg_churn:.2f}", "Average churn score")
    with kcols[3]:
        kpi_card("Avg Data", format_num(avg_data, " GB"), "Average monthly usage")
    with kcols[4]:
        kpi_card("Complaints", format_num(complaints), "Within selected months")
    with kcols[5]:
        kpi_card("Support", format_num(support), "Interaction count")

    chart_type = CHART_TYPES[chart_label]
    chart_tabs = st.tabs(["Revenue", "Risk", "Segments", "City", "Usage", "Customers"])

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
            usage,
            x="summary_month",
            y="value",
            color="metric",
            markers=True,
            title="Average usage trend",
            template=st.session_state.get("plot_template", "plotly_dark"),
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
            "Export filtered customer view",
            customer_view.to_csv(index=False).encode("utf-8"),
            file_name="customer_analytics_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Filtered dataset"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Export full filtered dataset",
            df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_customer_360_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("SQL visibility for analytics filters"):
        st.code(sql, language="sql")
        st.json({"params": params})


def show_customer_insights():
    hero(
        "Customer Insights",
        "Search, inspect, and export a complete customer profile with churn, value, subscription, billing, complaint, and support context.",
        "Single customer view",
    )

    search = st.text_input(
        "Search customer",
        placeholder="Enter customer ID, name, phone, city, or email",
    )
    if search.strip():
        like = f"%{search.strip()}%"
        candidates = query_df(
            """
            SELECT customer_id, full_name, city, customer_segment, phone_number, email
            FROM customers
            WHERE CAST(customer_id AS TEXT) LIKE ?
               OR full_name LIKE ?
               OR phone_number LIKE ?
               OR email LIKE ?
               OR city LIKE ?
            ORDER BY customer_id
            LIMIT 100
            """,
            (like, like, like, like, like),
        )
    else:
        candidates = query_df(
            """
            SELECT customer_id, full_name, city, customer_segment, phone_number, email
            FROM customers
            ORDER BY customer_id
            LIMIT 100
            """
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
        """
        SELECT
            c.*,
            ch.churn_score,
            ch.risk_level,
            ch.main_risk_reason,
            ch.recommended_action,
            vs.value_segment,
            vs.arpu_jod,
            vs.total_revenue_6m_jod,
            vs.lifetime_months
        FROM customers c
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments vs ON vs.customer_id = c.customer_id
        WHERE c.customer_id = ?
        """,
        (customer_id,),
    ).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Customer", customer["full_name"], f"ID {customer_id}")
    with c2:
        kpi_card("Risk Level", customer.get("risk_level", "N/A"), f"Score {customer.get('churn_score', 0):.2f}")
    with c3:
        kpi_card("Value Segment", customer.get("value_segment", "N/A"), format_num(customer.get("arpu_jod", 0), " JOD ARPU"))
    with c4:
        kpi_card("6M Revenue", format_num(customer.get("total_revenue_6m_jod", 0), " JOD"), f"{customer.get('lifetime_months', 0)} months lifetime")

    st.markdown(
        f"""
        <div class="shell-card">
          <span class="insight-tag">{customer.get("customer_segment", "Segment")}</span>
          <span class="insight-tag">{customer.get("city", "City")}</span>
          <span class="insight-tag">{customer.get("preferred_language", "Language")}</span>
          <span class="insight-tag">{customer.get("customer_status", customer.get("status", "Status"))}</span>
          <h3 style="margin-top:.6rem;">Recommended action</h3>
          <p class="muted">{customer.get("recommended_action", "No recommended action available.")}</p>
          <h3>Main risk reason</h3>
          <p class="muted">{customer.get("main_risk_reason", "No risk reason available.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Profile", "Subscriptions", "Billing", "Complaints", "Support", "Monthly usage", "Ask AI"])

    with tabs[0]:
        profile_df = customer.to_frame(name="value").reset_index().rename(columns={"index": "field"})
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Export profile",
            profile_df.to_csv(index=False).encode("utf-8"),
            file_name=f"customer_{customer_id}_profile.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tabs[1]:
        subs = query_df(
            """
            SELECT s.subscription_id, s.msisdn, s.service_type, s.activation_date, s.contract_end_date,
                   s.status, s.auto_renewal_flag, s.primary_subscription_flag,
                   p.plan_name, p.plan_category, p.monthly_fee_jod, p.technology
            FROM subscriptions s
            LEFT JOIN plans p ON p.plan_id = s.plan_id
            WHERE s.customer_id = ?
            ORDER BY s.primary_subscription_flag DESC, s.activation_date DESC
            """,
            (customer_id,),
        )
        st.dataframe(subs, use_container_width=True, hide_index=True)

    with tabs[2]:
        billing = query_df(
            """
            SELECT i.invoice_id, i.issue_date, i.due_date, i.total_amount_jod,
                   i.payment_status, i.days_overdue, a.account_number, a.account_type
            FROM invoices i
            JOIN accounts a ON a.account_id = i.account_id
            WHERE a.customer_id = ?
            ORDER BY i.issue_date DESC
            LIMIT 50
            """,
            (customer_id,),
        )
        st.dataframe(billing, use_container_width=True, hide_index=True)

    with tabs[3]:
        complaints = query_df(
            """
            SELECT complaint_id, complaint_date, complaint_category, severity, status,
                   resolved_date, compensation_amount_jod, complaint_description
            FROM complaints
            WHERE customer_id = ?
            ORDER BY complaint_date DESC
            LIMIT 50
            """,
            (customer_id,),
        )
        st.dataframe(complaints, use_container_width=True, hide_index=True)

    with tabs[4]:
        support = query_df(
            """
            SELECT interaction_id, interaction_datetime, channel, reason_category, issue_type,
                   priority, resolution_status, resolution_time_minutes, customer_sentiment
            FROM support_interactions
            WHERE customer_id = ?
            ORDER BY interaction_datetime DESC
            LIMIT 50
            """,
            (customer_id,),
        )
        st.dataframe(support, use_container_width=True, hide_index=True)

    with tabs[5]:
        monthly = query_df(
            """
            SELECT summary_month, total_revenue_jod, voice_minutes, data_used_gb, sms_count,
                   support_interactions_count, complaints_count, payment_delay_days, churn_score
            FROM customer_monthly_summary
            WHERE customer_id = ?
            ORDER BY summary_month
            """,
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
                template=st.session_state.get("plot_template", "plotly_dark"),
            )
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
            st.dataframe(monthly, use_container_width=True, hide_index=True)

    with tabs[6]:
        suggested = f"Show me the full profile, plan, complaints, churn risk, and recommended action for customer {customer_id}."
        st.code(suggested)
        if st.button("Send this question to AI Chat", type="primary", use_container_width=True):
            st.session_state.pending_prompt = suggested
            st.session_state.page = "Chat"
            st.rerun()


def show_chart_builder():
    hero(
        "Chart Builder",
        "Describe the chart you want in business language. The app plans a safe read-only query and turns the result into a visual.",
        "Natural-language visuals",
    )

    with st.container(border=True):
        question = st.text_area(
            "Chart inquiry",
            value="Build a chart based on customer with ID = 9 by their complaints type and number.",
            height=120,
        )
        cols = st.columns([1, 1, 2])
        with cols[0]:
            chart_label = st.selectbox("Chart type", list(CHART_TYPES.keys()))
        with cols[1]:
            run = st.button("Create Chart", type="primary", use_container_width=True)
        with cols[2]:
            st.caption("Tip: ask for one clear chart, for example conversion by campaign, churn by city, or complaints by category.")

    if run:
        with st.spinner("Building chart from database..."):
            st.session_state.last_chart = build_chart_from_question(question, CHART_TYPES[chart_label])

    if st.session_state.last_chart:
        shell_start()
        render_chart(st.session_state.last_chart)
        shell_end()


def show_sql_workspace():
    hero(
        "SQL Workspace",
        "A clean read-only SQL workspace for analysts. It preserves safety by allowing SELECT queries only.",
        "Safe query runner",
    )

    templates = {
        "Total customers": "SELECT COUNT(*) AS total_customers FROM customers",
        "Top churn customers": """
SELECT c.customer_id, c.full_name, c.city, c.customer_segment, ch.churn_score, ch.risk_level, ch.main_risk_reason
FROM customer_churn_scores ch
JOIN customers c ON c.customer_id = ch.customer_id
ORDER BY ch.churn_score DESC
LIMIT 10
""".strip(),
        "Revenue by segment": """
SELECT vs.value_segment, COUNT(*) AS customers, ROUND(AVG(vs.arpu_jod), 2) AS avg_arpu, ROUND(SUM(vs.total_revenue_6m_jod), 2) AS revenue_6m
FROM customer_value_segments vs
GROUP BY vs.value_segment
ORDER BY revenue_6m DESC
""".strip(),
        "Open complaints": """
SELECT complaint_category, severity, status, COUNT(*) AS total
FROM complaints
WHERE status != 'Resolved'
GROUP BY complaint_category, severity, status
ORDER BY total DESC
""".strip(),
    }

    selected_template = st.selectbox("Query template", list(templates.keys()))
    if st.button("Load template", use_container_width=True):
        st.session_state["standalone_sql_editor"] = templates[selected_template]
        st.rerun()

    render_sql_runner(templates[selected_template], key_prefix="standalone")


def show_suggested_questions():
    hero(
        "Prompt Library",
        "Use ready-made business prompts to generate database-backed answers faster.",
        "Suggested workflows",
    )
    cols = st.columns(3)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="prompt-card">
                  <b>Use case {i + 1}</b>
                  <p>{question}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Send to Chat", key=f"suggested_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.session_state.page = "Chat"
                st.rerun()


def show_data_catalog():
    hero(
        "Data Catalog",
        "Browse the SQLite Customer 360 data model, table sizes, and field definitions before writing queries.",
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
        st.subheader(f"{selected} columns")
        st.dataframe(cols_df[["name", "type", "notnull", "pk"]], use_container_width=True, hide_index=True)
        st.subheader("Sample rows")
        sample = query_df(f'SELECT * FROM "{selected}" LIMIT 10')
        st.dataframe(sample, use_container_width=True, hide_index=True)

    with st.expander("Export catalog"):
        st.download_button(
            "Download table inventory",
            inventory.to_csv(index=False).encode("utf-8"),
            file_name="data_catalog.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-card">
              <div class="brand-title">Customer 360 AI Copilot</div>
              <div class="brand-copy">
                Premium analytics workspace for customers, churn, revenue, complaints, support, campaigns, and network signals.
              </div>
              <div class="chip-row">
                <span class="chip">SQL-backed</span>
                <span class="chip">AI chat</span>
                <span class="chip">Dynamic BI</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.theme_mode = "Dark"

        st.markdown('<div class="side-label">Current workspace</div>', unsafe_allow_html=True)
        active_item = next((item for item in NAV_ITEMS if item[0] == st.session_state.page), NAV_ITEMS[0])
        st.markdown(
            f"""
            <div class="active-page">
              {active_item[2]} {active_item[1]}
              <small>{active_item[3]}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("＋ New Chat", type="primary", use_container_width=True):
            create_new_chat()
            st.rerun()

        st.markdown('<div class="side-label">Saved chats</div>', unsafe_allow_html=True)
        for chat in st.session_state.chat_sessions[:8]:
            label = "💬 " + chat["title"]
            if st.button(label, key=f"select_{chat['id']}", use_container_width=True):
                st.session_state.current_chat_id = chat["id"]
                st.session_state.page = "Chat"
                st.rerun()

        if st.button("Delete current chat", use_container_width=True):
            delete_current_chat()
            st.rerun()

        st.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
        for page, title, icon, _desc in NAV_ITEMS:
            if st.button(f"{icon} {title}", key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()


def main():
    ensure_state()
    inject_css()
    render_sidebar()

    page = st.session_state.page
    if page == "Chat":
        show_chat()
    elif page == "Analytics":
        show_dynamic_analytics()
    elif page == "Customer Insights":
        show_customer_insights()
    elif page == "Chart Builder":
        show_chart_builder()
    elif page == "SQL Query Builder":
        show_sql_workspace()
    elif page == "Suggested Questions":
        show_suggested_questions()
    elif page == "Data Catalog":
        show_data_catalog()
    else:
        show_chat()


if __name__ == "__main__":
    main()
