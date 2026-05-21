import json
import os
import re
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "zain_customer_360_ai_demo.db"
QA_MEMORY_PATH = BASE_DIR / "rag_qa_memory.txt"
MODEL_NAME = "gpt-4.1-mini"
TOP_K = 5
RAG_MIN_SIMILARITY = 0.52
NON_DATABASE_PATTERNS = (
    r"^\s*(hi|hello|hey|مرحبا|اهلا|أهلا|السلام عليكم)\s*[!.؟?]*\s*$",
    r"^\s*(thanks|thank you|شكرا|شكرًا)\s*[!.؟?]*\s*$",
    r"^\s*(how are you|كيفك|كيف الحال)\s*[!.؟?]*\s*$",
)


def load_openai_key():
    if os.environ.get("OPENAI_API_KEY"):
        return

    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENAI_API_KEY" and value.strip():
            os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
            return


def extract_final_text(result):
    last_message = result["messages"][-1]

    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        return last_message.content

    if hasattr(last_message, "content_blocks"):
        parts = []
        for block in last_message.content_blocks:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
                elif "content" in block:
                    parts.append(str(block["content"]))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)

    return str(last_message)


def _connect():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_markdown(rows):
    if not rows:
        return "No matching rows found."

    headers = rows[0].keys()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    return "\n".join(lines)


def _run_rows(query, params=()):
    with _connect() as conn:
        return conn.execute(query, params).fetchall()


def _normalize_question(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _question_tokens(text):
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "by",
        "for",
        "from",
        "have",
        "how",
        "in",
        "is",
        "me",
        "of",
        "on",
        "or",
        "show",
        "the",
        "to",
        "what",
        "which",
        "with",
    }
    tokens = set()
    for token in re.findall(r"[a-z0-9_]+", _normalize_question(text)):
        if len(token) <= 1 or token in stop_words:
            continue
        tokens.add(token)
        if len(token) > 3 and token.endswith("s"):
            tokens.add(token[:-1])
    return tokens


def _number_tokens(text):
    return set(re.findall(r"\b\d+\b", str(text)))


def _question_similarity(left, right):
    left_norm = _normalize_question(left)
    right_norm = _normalize_question(right)
    if not left_norm or not right_norm:
        return 0.0

    left_numbers = _number_tokens(left_norm)
    right_numbers = _number_tokens(right_norm)
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return 0.0

    left_tokens = _question_tokens(left_norm)
    right_tokens = _question_tokens(right_norm)
    token_score = 0.0
    if left_tokens or right_tokens:
        token_score = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    sequence_score = SequenceMatcher(None, left_norm, right_norm).ratio()
    return (0.6 * token_score) + (0.4 * sequence_score)


def _read_qa_memory():
    if not QA_MEMORY_PATH.exists():
        return []

    records = []
    for line in QA_MEMORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("question") and record.get("answer"):
            records.append(record)
    return records


def retrieve_rag_answer(question):
    q = _normalize_question(question)
    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in NON_DATABASE_PATTERNS):
        return None

    best_record = None
    best_score = 0.0
    for record in _read_qa_memory():
        score = _question_similarity(question, record.get("question", ""))
        if score > best_score:
            best_score = score
            best_record = record

    if not best_record or best_score < RAG_MIN_SIMILARITY:
        return None

    return {
        "answer": best_record["answer"],
        "sql": "",
        "source": "RAG Agent",
        "matched_question": best_record.get("question", ""),
        "match_score": round(best_score, 3),
    }


def combine_agent_payloads(sql_payload, rag_payload):
    sql_answer = str(sql_payload.get("answer", "")).strip()
    rag_answer = str(rag_payload.get("answer", "")).strip()
    matched_question = rag_payload.get("matched_question", "")
    match_score = rag_payload.get("match_score", "")

    if _normalize_question(sql_answer) == _normalize_question(rag_answer):
        answer = sql_answer
    else:
        memory_note = ""
        if matched_question:
            memory_note = f"\n\nMemory Context\nMatched prior question: {matched_question}"
            if match_score != "":
                memory_note += f" (similarity {match_score})"
        answer = (
            f"{sql_answer}\n\n"
            "Related Memory Answer\n"
            f"{rag_answer}"
            f"{memory_note}"
        ).strip()

    return {
        "answer": answer,
        "sql": sql_payload.get("sql", ""),
        "source": "SQL Agent + RAG Agent",
        "matched_question": matched_question,
        "match_score": match_score,
    }


def save_qa_to_memory(question, answer, sql="", source="SQL Agent"):
    question = str(question).strip()
    answer = str(answer).strip()
    if not question or not answer:
        return

    q = _normalize_question(question)
    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in NON_DATABASE_PATTERNS):
        return

    for record in _read_qa_memory():
        if _normalize_question(record.get("question", "")) == q:
            return

    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_agent": source,
        "question": question,
        "answer": answer,
        "sql": str(sql or "").strip(),
    }
    with QA_MEMORY_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _schema_summary():
    with _connect() as conn:
        table_names = [
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        ]
        parts = []
        for table in table_names:
            columns = [f"{row['name']} {row['type']}" for row in conn.execute(f"PRAGMA table_info({table})")]
            parts.append(f"{table}: " + ", ".join(columns))
        return "\n".join(parts)


def _extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("The chart agent did not return valid JSON.")
    return json.loads(text[start : end + 1])


def _validate_chart_sql(sql):
    normalized = sql.strip().rstrip(";")
    lowered = normalized.lower()
    banned = ("insert", "update", "delete", "drop", "alter", "truncate", "create", "replace", "attach", "detach")
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT chart queries are allowed.")
    if any(re.search(rf"\b{word}\b", lowered) for word in banned):
        raise ValueError("The chart request tried to use a non-read-only query.")
    return normalized


def validate_read_only_sql(sql):
    normalized = sql.strip().rstrip(";")
    lowered = normalized.lower()
    banned = ("insert", "update", "delete", "drop", "alter", "truncate", "create", "replace", "attach", "detach", "pragma")
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT queries are allowed.")
    if any(re.search(rf"\b{word}\b", lowered) for word in banned):
        raise ValueError("Only read-only SELECT queries are allowed.")
    return normalized


def execute_sql_query(sql, limit=50):
    sql = validate_read_only_sql(sql)
    limited_sql = sql
    if " limit " not in f" {sql.lower()} ":
        limited_sql = f"{sql}\nLIMIT {max(1, min(int(limit), 100))}"
    rows = [dict(row) for row in _run_rows(limited_sql)]
    columns = list(rows[0].keys()) if rows else []
    return {"columns": columns, "rows": rows, "sql": limited_sql}


def get_database_overview():
    with _connect() as conn:
        table_rows = [
            {"label": row["name"], "value": conn.execute(f"SELECT COUNT(*) AS count FROM {row['name']}").fetchone()["count"]}
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        ]

    kpis = [
        {"label": "Customers", "value": _run_rows("SELECT COUNT(*) AS value FROM customers")[0]["value"]},
        {"label": "Complaints", "value": _run_rows("SELECT COUNT(*) AS value FROM complaints")[0]["value"]},
        {"label": "Campaign Responses", "value": _run_rows("SELECT COUNT(*) AS value FROM customer_campaign_responses")[0]["value"]},
        {"label": "Network Events", "value": _run_rows("SELECT COUNT(*) AS value FROM network_events")[0]["value"]},
    ]
    charts = [
        {
            "title": "Customers by Segment",
            "metric": "Customers",
            "chart_type": "pie",
            "rows": [dict(row) for row in _run_rows("SELECT customer_segment AS label, COUNT(*) AS value FROM customers GROUP BY customer_segment ORDER BY value DESC")],
        },
        {
            "title": "Customers by City",
            "metric": "Customers",
            "chart_type": "bar",
            "rows": [dict(row) for row in _run_rows("SELECT city AS label, COUNT(*) AS value FROM customers GROUP BY city ORDER BY value DESC LIMIT 8")],
        },
        {
            "title": "Churn Risk Levels",
            "metric": "Customers",
            "chart_type": "pie",
            "rows": [dict(row) for row in _run_rows("SELECT risk_level AS label, COUNT(*) AS value FROM customer_churn_scores GROUP BY risk_level ORDER BY value DESC")],
        },
        {
            "title": "Largest Tables",
            "metric": "Rows",
            "chart_type": "bar",
            "rows": sorted(table_rows, key=lambda row: row["value"], reverse=True)[:10],
        },
    ]
    return {
        "kpis": kpis,
        "charts": charts,
        "tables": sorted(table_rows, key=lambda row: row["label"]),
        "summary": "This overview highlights database size, customer distribution, churn risk, and major activity tables.",
    }


def _run_dynamic_chart_query(question, chart_type, limit):
    load_openai_key()
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "title": "Chart unavailable",
            "metric": "No data",
            "chart_type": chart_type,
            "rows": [],
            "summary": "OPENAI_API_KEY is not set, so the dynamic chart query could not be generated.",
        }

    model = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
    prompt = f"""
You are a chart SQL planner for a SQLite telecom Customer 360 database.

Database schema:
{_schema_summary()}

User chart request:
{question}

Return only one JSON object with these fields:
- title: short chart title
- metric: short label for the numeric metric
- chart_type: "bar", "horizontal_bar", "pie", "doughnut", "line", or "area"
- sql: one safe SQLite SELECT query
- summary: one short business explanation

Rules:
- The SQL must be read-only SELECT only.
- The SQL must return exactly two required aliases: label and value.
- label must be human-readable text.
- value must be numeric.
- Use joins when needed.
- If the request asks about a specific customer, filter by that customer_id.
- Limit the chart to at most {limit} rows unless grouping naturally returns fewer.
- If the requested data is not available in the schema, set sql to an empty string and explain that in summary.
"""
    result = model.invoke([{"role": "user", "content": prompt}])
    content = result.content if hasattr(result, "content") else str(result)
    spec = _extract_json_object(content)
    sql = str(spec.get("sql", "")).strip()
    if not sql:
        return {
            "title": spec.get("title") or "Chart unavailable",
            "metric": spec.get("metric") or "No data",
            "chart_type": chart_type,
            "rows": [],
            "summary": spec.get("summary") or "The requested chart cannot be created from the available database fields.",
        }

    sql = _validate_chart_sql(sql)
    rows = [dict(row) for row in _run_rows(sql)]
    normalized_rows = []
    for row in rows[:limit]:
        if "label" not in row or "value" not in row:
            continue
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue
        normalized = dict(row)
        normalized["label"] = str(row["label"])
        normalized["value"] = value
        normalized_rows.append(normalized)

    return {
        "title": spec.get("title") or "Custom Database Chart",
        "metric": spec.get("metric") or "Value",
        "chart_type": chart_type if chart_type in {"bar", "horizontal_bar", "pie", "doughnut", "line", "area"} else spec.get("chart_type", "bar"),
        "rows": normalized_rows,
        "summary": spec.get("summary") or "Chart created from the database.",
    }


def answer_direct_query(question):
    q = question.lower()
    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in NON_DATABASE_PATTERNS):
        return {
            "answer": "Hello. Ask me a question about customers, churn, complaints, billing, campaigns, network events, or anything in the Customer 360 database.",
            "sql": "",
        }

    limit_match = re.search(r"\btop\s+(\d+)|\blimit\s+(\d+)", q)
    limit = int(next(value for value in limit_match.groups() if value)) if limit_match else 10
    limit = max(1, min(limit, 25))
    customer_match = re.search(r"\bcustomer(?:\s+with)?\s+id\s*=?\s*(\d+)|\bcustomer_id\s*=?\s*(\d+)|\bid\s*=\s*(\d+)", q)
    customer_id = int(next(value for value in customer_match.groups() if value)) if customer_match else None

    if "how many" in q and "customer" in q:
        sql = "SELECT COUNT(*) AS total_customers FROM customers"
        total = _run_rows(sql)[0]["total_customers"]
        answer = (
            f"Direct Answer\nThere are {total:,} customers in the database.\n\n"
            f"Key Numbers\n- Total customers: {total:,}\n\n"
            "Business Interpretation\nThis is the current Customer 360 base available for segmentation, churn, revenue, billing, and network analysis."
        )
        return {"answer": answer, "sql": sql}

    if "churn" in q and ("highest" in q or "top" in q or "risk" in q):
        sql = """
SELECT
    c.customer_id,
    c.full_name,
    c.city,
    c.customer_segment,
    ch.churn_score,
    ch.risk_level,
    ch.main_risk_reason,
    ch.recommended_action
FROM customer_churn_scores ch
JOIN customers c ON ch.customer_id = c.customer_id
ORDER BY ch.churn_score DESC
LIMIT ?
""".strip()
        rows = _run_rows(
            sql,
            (limit,),
        )
        answer = (
            f"Direct Answer\nThese are the top {limit} customers with the highest churn scores.\n\n"
            f"{_rows_to_markdown(rows)}\n\n"
            "Business Interpretation\nCustomers at the top of this list should be prioritized because their churn scores and risk reasons indicate a higher likelihood of leaving.\n\n"
            "Recommended Next Action\nContact them with retention offers matched to the recommended action and monitor complaint or service-quality signals."
        )
        return {"answer": answer, "sql": sql.replace("?", str(limit))}

    if "complaint" in q:
        if customer_id is not None:
            sql = """
SELECT
    complaint_category,
    COUNT(*) AS total_complaints
FROM complaints
WHERE customer_id = ?
GROUP BY complaint_category
ORDER BY total_complaints DESC
""".strip()
            rows = _run_rows(sql, (customer_id,))
            customer = _run_rows("SELECT full_name FROM customers WHERE customer_id = ?", (customer_id,))
            if not customer:
                return {
                    "answer": f"Direct Answer\nNo customer was found with customer_id = {customer_id}.",
                    "sql": f"SELECT full_name FROM customers WHERE customer_id = {customer_id}",
                }
            customer_name = customer[0]["full_name"]
            if not rows:
                return {
                    "answer": f"Direct Answer\nNo complaint records were found for {customer_name} with customer_id = {customer_id}.",
                    "sql": sql.replace("?", str(customer_id)),
                }
            return {
                "answer": f"Direct Answer\nComplaint types for {customer_name} with customer_id = {customer_id}:\n\n" + _rows_to_markdown(rows),
                "sql": sql.replace("?", str(customer_id)),
            }

        sql = """
SELECT complaint_category, COUNT(*) AS total_complaints
FROM complaints
GROUP BY complaint_category
ORDER BY total_complaints DESC
LIMIT ?
""".strip()
        rows = _run_rows(sql, (limit,))
        return {"answer": "Direct Answer\nTop complaint categories:\n\n" + _rows_to_markdown(rows), "sql": sql.replace("?", str(limit))}

    if "campaign" in q and ("conversion" in q or "best" in q):
        sql = """
SELECT
    ca.campaign_name,
    ca.campaign_type,
    ca.target_segment,
    COUNT(cr.response_id) AS total_sent,
    SUM(cr.converted_flag) AS total_converted,
    ROUND(100.0 * SUM(cr.converted_flag) / COUNT(cr.response_id), 2) AS conversion_rate_percent
FROM campaigns ca
JOIN customer_campaign_responses cr ON ca.campaign_id = cr.campaign_id
GROUP BY ca.campaign_id, ca.campaign_name, ca.campaign_type, ca.target_segment
ORDER BY conversion_rate_percent DESC
LIMIT ?
""".strip()
        rows = _run_rows(
            sql,
            (limit,),
        )
        return {"answer": "Direct Answer\nBest campaigns by conversion rate:\n\n" + _rows_to_markdown(rows), "sql": sql.replace("?", str(limit))}

    if "city" in q and "customer" in q:
        sql = """
SELECT city, COUNT(*) AS total_customers
FROM customers
GROUP BY city
ORDER BY total_customers DESC
LIMIT ?
""".strip()
        rows = _run_rows(sql, (limit,))
        return {"answer": "Direct Answer\nCustomer distribution by city:\n\n" + _rows_to_markdown(rows), "sql": sql.replace("?", str(limit))}

    if "segment" in q and "revenue" in q:
        sql = """
SELECT
    value_segment,
    COUNT(*) AS total_customers,
    ROUND(AVG(arpu_jod), 2) AS avg_arpu,
    ROUND(AVG(total_revenue_6m_jod), 2) AS avg_revenue_6m
FROM customer_value_segments
GROUP BY value_segment
ORDER BY avg_revenue_6m DESC
LIMIT ?
""".strip()
        rows = _run_rows(
            sql,
            (limit,),
        )
        return {"answer": "Direct Answer\nRevenue by value segment:\n\n" + _rows_to_markdown(rows), "sql": sql.replace("?", str(limit))}

    if "network" in q or "affected" in q:
        sql = """
SELECT nt.city, SUM(ne.affected_customers) AS total_affected_customers
FROM network_events ne
JOIN network_towers nt ON ne.tower_id = nt.tower_id
GROUP BY nt.city
ORDER BY total_affected_customers DESC
LIMIT ?
""".strip()
        rows = _run_rows(sql, (limit,))
        return {"answer": "Direct Answer\nCities with highest network impact:\n\n" + _rows_to_markdown(rows), "sql": sql.replace("?", str(limit))}

    return None


def build_chart_from_question(question, chart_type="bar"):
    q = question.lower()
    limit_match = re.search(r"\btop\s+(\d+)|\blimit\s+(\d+)", q)
    limit = int(next(value for value in limit_match.groups() if value)) if limit_match else 10
    limit = max(1, min(limit, 15))
    customer_match = re.search(r"\bcustomer(?:\s+with)?\s+id\s*=?\s*(\d+)|\bcustomer_id\s*=?\s*(\d+)", q)
    customer_id = int(next(value for value in customer_match.groups() if value)) if customer_match else None

    chart = chart_type if chart_type in {"bar", "horizontal_bar", "pie", "doughnut", "line", "area"} else "bar"

    if "churn" in q:
        rows = _run_rows(
            """
            SELECT
                c.full_name AS label,
                ROUND(ch.churn_score, 2) AS value,
                c.customer_id,
                c.city,
                c.customer_segment,
                ch.risk_level,
                ch.main_risk_reason
            FROM customer_churn_scores ch
            JOIN customers c ON ch.customer_id = c.customer_id
            ORDER BY ch.churn_score DESC
            LIMIT ?
            """,
            (limit,),
        )
        return {
            "title": f"Top {limit} Customers by Churn Score",
            "metric": "Churn score",
            "chart_type": chart,
            "rows": [dict(row) for row in rows],
            "summary": "The chart ranks customers by churn score. Higher scores indicate customers needing faster retention attention.",
        }

    if "complaint" in q:
        if customer_id is not None:
            rows = _run_rows(
                """
                SELECT complaint_category AS label, COUNT(*) AS value
                FROM complaints
                WHERE customer_id = ?
                GROUP BY complaint_category
                ORDER BY value DESC
                """,
                (customer_id,),
            )
            customer = _run_rows("SELECT full_name FROM customers WHERE customer_id = ?", (customer_id,))
            customer_name = customer[0]["full_name"] if customer else f"Customer {customer_id}"
            return {
                "title": f"Complaint Types for {customer_name}",
                "metric": "Complaints",
                "chart_type": chart,
                "rows": [dict(row) for row in rows],
                "summary": (
                    f"This chart shows complaint volume by complaint type for customer ID {customer_id}."
                    if rows
                    else f"No complaint records were found for customer ID {customer_id}."
                ),
            }

        rows = _run_rows(
            """
            SELECT complaint_category AS label, COUNT(*) AS value
            FROM complaints
            GROUP BY complaint_category
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )
        return {"title": "Top Complaint Categories", "metric": "Complaints", "chart_type": chart, "rows": [dict(row) for row in rows], "summary": "The chart highlights the complaint themes customers raise most often."}

    if "campaign" in q:
        rows = _run_rows(
            """
            SELECT ca.campaign_name AS label, ROUND(100.0 * SUM(cr.converted_flag) / COUNT(cr.response_id), 2) AS value
            FROM campaigns ca
            JOIN customer_campaign_responses cr ON ca.campaign_id = cr.campaign_id
            GROUP BY ca.campaign_id, ca.campaign_name
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )
        return {"title": "Best Campaign Conversion Rates", "metric": "Conversion %", "chart_type": chart, "rows": [dict(row) for row in rows], "summary": "The chart compares campaigns by conversion rate."}

    if "network" in q or "affected" in q:
        rows = _run_rows(
            """
            SELECT nt.city AS label, SUM(ne.affected_customers) AS value
            FROM network_events ne
            JOIN network_towers nt ON ne.tower_id = nt.tower_id
            GROUP BY nt.city
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )
        return {"title": "Network Impact by City", "metric": "Affected customers", "chart_type": chart, "rows": [dict(row) for row in rows], "summary": "The chart shows where network events affected the most customers."}

    if "segment" in q and "revenue" in q:
        rows = _run_rows(
            """
            SELECT value_segment AS label, ROUND(AVG(total_revenue_6m_jod), 2) AS value
            FROM customer_value_segments
            GROUP BY value_segment
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )
        return {"title": "Average 6M Revenue by Segment", "metric": "JOD/customer", "chart_type": chart, "rows": [dict(row) for row in rows], "summary": "The chart compares value segments by average six-month revenue."}

    if "city" in q and "customer" in q:
        rows = _run_rows(
            """
            SELECT city AS label, COUNT(*) AS value
            FROM customers
            GROUP BY city
            ORDER BY value DESC
            LIMIT ?
            """,
            (limit,),
        )
        return {"title": "Customers by City", "metric": "Customers", "chart_type": chart, "rows": [dict(row) for row in rows], "summary": "The chart shows the largest customer concentrations by city."}

    try:
        return _run_dynamic_chart_query(question, chart, limit)
    except Exception as exc:
        return {
            "title": "Chart unavailable",
            "metric": "No data",
            "chart_type": chart,
            "rows": [],
            "summary": f"I could not create this chart from the available database fields. Details: {type(exc).__name__}: {exc}",
        }


def create_sql_agent():
    load_openai_key()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH.as_posix()}")
    model = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    system_prompt = f"""
You are a professional telecom Business Intelligence SQL Agent for Zain Jordan.

You are connected to a SQLite Customer 360 telecom database that contains customer, subscription, billing, payment, revenue, complaint, support, campaign, churn, network, plan, device, and usage data.

Your main role is to answer business questions accurately using safe SQL and explain the results in clear telecom business language.

Core Responsibilities:
- Understand the user's business question.
- Identify the correct business domain: customer, churn, revenue, billing, complaints, support, campaigns, network, plans, subscriptions, payments, usage, or customer profile.
- Inspect the available tables and schemas before writing SQL.
- Select only the tables and columns needed to answer the question.
- Generate syntactically correct SQLite queries.
- Double-check the SQL query before execution.
- Execute the query only after validating that it is safe and relevant.
- Explain the result in simple, practical business terms.
- Recommend an action when the result can support a business decision.

Strict Safety Rules:
- Use SELECT queries only.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, REPLACE, ATTACH, DETACH, or PRAGMA.
- Never modify the database.
- Never expose unnecessary data.
- Never query all columns using SELECT * unless absolutely required.
- Use relevant columns only.
- Apply a LIMIT of at most {TOP_K} rows unless the user clearly requests a specific number or the query returns aggregated results.
- If the user asks for “top”, “highest”, “lowest”, “best”, or “worst”, always use ORDER BY with the correct metric.
- If the user asks about a specific customer, filter by customer_id when provided.
- If the question is ambiguous, state your assumption clearly before answering.
- If the requested table or column does not exist, inspect the schema and correct the query.
- Do not guess facts that are not available in the database.
- Do not provide unsupported conclusions.
- If the database does not contain enough data to answer, say so clearly.

Agent Behavior:
- Be reliable, concise, and business-focused.
- Prioritize accuracy over long explanations.
- Use joins only when needed.
- Prefer aggregated results for business questions.
- Use readable column aliases.
- Format numeric values clearly.
- Use JOD for revenue, payments, invoices, ARPU, and financial amounts when applicable.
- Use percentages for conversion rates, churn rates, and ratios when applicable.
- For customer lists, include useful identifiers such as customer_id, full_name, city, segment, risk level, and recommended action when available.
- For operational issues, highlight priority, severity, status, and affected customers when available.
- For repeated or similar questions, keep the answer consistent with the data.

Telecom Business Context:
- Churn analysis usually involves customers and customer_churn_scores.
- Revenue and value analysis usually involves customer_value_segments, customer_monthly_summary, invoices, payments, transactions, subscriptions, and plans.
- Complaint analysis usually involves complaints and support_interactions.
- Campaign analysis usually involves campaigns and customer_campaign_responses.
- Network analysis usually involves network_towers and network_events.
- Subscription and plan analysis usually involves subscriptions and plans.
- Billing analysis usually involves accounts, invoices, payments, transactions, and topups.
- Customer 360 profile questions usually require combining customers, subscriptions, plans, complaints, support_interactions, invoices, customer_churn_scores, and customer_value_segments.
- Usage analysis usually involves customer_monthly_summary and subscription-level usage fields if available.

Common Use Cases You Should Support:
- Identify customers with the highest churn risk.
- Explain main churn risk reasons.
- Find high-value customers with negative sentiment or complaints.
- Analyze customer distribution by city, segment, service type, or value segment.
- Compare revenue by customer segment, plan, city, or service type.
- Find overdue invoices and payment issues.
- Analyze complaint categories, unresolved complaints, and repeated complaints.
- Summarize support interactions by channel, reason, sentiment, and priority.
- Measure campaign conversion rate and generated revenue.
- Identify network events affecting the highest number of customers.
- Build customer-level summaries with profile, plan, complaints, churn risk, and recommended action.
- Support dashboard and chart requests using clean aggregated outputs.

SQL Quality Rules:
- Always choose the most direct query that answers the question.
- Use COUNT, SUM, AVG, MIN, MAX, ROUND, GROUP BY, ORDER BY, and LIMIT where appropriate.
- Use INNER JOIN when records must match across tables.
- Use LEFT JOIN when the user needs customers even if related records are missing.
- Avoid unnecessary joins.
- Avoid returning raw records when an aggregated answer is better.
- For date or month filters, use the available date/month columns in the schema.
- For unresolved complaints, filter out resolved statuses when the status field exists.
- For conversion rate, calculate:
  ROUND(100.0 * SUM(converted_flag) / COUNT(*), 2)
  when converted_flag is available.
- For churn prioritization, order by churn_score DESC.
- For revenue prioritization, order by total revenue or average revenue DESC.
- For network impact, order by affected_customers DESC or SUM(affected_customers) DESC.

Answer Format:
Always respond using this structure when possible:

1. Direct Answer
- Give the main answer clearly and immediately.

2. Key Numbers
- Provide the most important figures, rankings, totals, averages, or percentages.

3. Business Interpretation
- Explain what the result means from a telecom business perspective.

4. Recommended Next Action
- Provide a practical next step when useful.
- If no action is needed, omit this section.

5. SQL Used
- Include the SQL query used when available.

Important:
- Do not invent business insights.
- Do not over-explain.
- Do not provide generic advice that is not connected to the query result.
- Keep the response professional, accurate, and useful for telecom business users.
"""

    return create_agent(model=model, tools=sql_tools, system_prompt=system_prompt)


_SQL_AGENT = None


def plan_sql_for_question(question):
    q = question.lower()
    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in NON_DATABASE_PATTERNS):
        return ""

    load_openai_key()
    if not os.environ.get("OPENAI_API_KEY"):
        return ""

    model = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
    prompt = f"""
You are a SQLite SQL planner for a telecom Customer 360 database.

Database schema:
{_schema_summary()}

Question:
{question}

Return only one JSON object:
{{"sql": "SELECT ..."}}

Rules:
- SQL must be read-only SELECT only.
- Use the same business intent as the question.
- Use joins when needed.
- Use relevant columns only.
- Limit detailed row results to 25 rows unless the query is grouped.
- If the needed data is not in the schema, return {{"sql": ""}}.
"""
    result = model.invoke([{"role": "user", "content": prompt}])
    content = result.content if hasattr(result, "content") else str(result)
    spec = _extract_json_object(content)
    sql = str(spec.get("sql", "")).strip()
    return validate_read_only_sql(sql) if sql else ""


def _ask_live_sql_agent_payload(question):
    global _SQL_AGENT
    if _SQL_AGENT is None:
        _SQL_AGENT = create_sql_agent()

    result = _SQL_AGENT.invoke({"messages": [{"role": "user", "content": question}]})
    answer = extract_final_text(result)
    try:
        sql = plan_sql_for_question(question)
    except Exception:
        sql = ""
    return {"answer": answer, "sql": sql, "source": "SQL Agent"}


def ask_sql_agent_payload(question):
    direct_payload = answer_direct_query(question)
    rag_payload = retrieve_rag_answer(question)

    if direct_payload:
        direct_payload["source"] = "SQL Agent"
        payload = combine_agent_payloads(direct_payload, rag_payload) if rag_payload else direct_payload
        save_qa_to_memory(
            question,
            payload.get("answer", ""),
            payload.get("sql", ""),
            payload["source"],
        )
        return payload

    if rag_payload:
        try:
            sql_payload = _ask_live_sql_agent_payload(question)
        except Exception:
            save_qa_to_memory(question, rag_payload.get("answer", ""), "", rag_payload["source"])
            return rag_payload

        payload = combine_agent_payloads(sql_payload, rag_payload)
        save_qa_to_memory(question, payload.get("answer", ""), payload.get("sql", ""), payload["source"])
        return payload

    payload = _ask_live_sql_agent_payload(question)
    save_qa_to_memory(question, payload.get("answer", ""), payload.get("sql", ""), payload["source"])
    return payload


def ask_sql_agent(question):
    return ask_sql_agent_payload(question)["answer"]
