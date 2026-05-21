# Zain Customer 360 Streamlit App

Streamlit deployment entry point:

```bash
streamlit run streamlit_app.py
```

## Required files

Keep these files in the app folder when uploading to Streamlit:

- `streamlit_app.py`
- `class3_sql_agent_backend.py`
- `zain_customer_360_ai_demo.db`
- `zain-logo.png`
- `requirements.txt`
- `.streamlit/config.toml`

## OpenAI API key

For Streamlit Community Cloud, add this in **App settings -> Secrets**:

```toml
OPENAI_API_KEY = "your-new-api-key"
```

Do not commit a real API key to GitHub. Locally, the app can also read `.env`.
