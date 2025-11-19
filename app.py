# app.py — FinSentry AI — FINAL VERSION (OpenAI version)
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import pdfplumber
import re
from io import StringIO
import os
import numpy as np  # NEW: for log_amount etc.

# === OpenAI client ===
from openai import OpenAI

# Recommended: use environment variable instead of hard-coding the key
# Set OPENAI_API_KEY in your environment before running the app.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="FinSentry AI", layout="wide")
st.title("FinSentry AI – Group 2 Final Demo")
st.success("Model loaded: data_1.joblib")

@st.cache_resource
def load_model():
    return joblib.load("data_1.joblib")

model = load_model()

def extract_text_smart(pdf_path: str) -> str:
    """
    Try to extract:
    1) normal text (page.extract_text)
    2) table content (page.extract_tables) as CSV-like rows

    Returns a big text string that will be sent to the LLM.
    """
    text_parts = []
    table_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 1) Raw text
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

            # 2) Tables → turn each row into a comma-separated line
            try:
                tables = page.extract_tables()
                for table in tables or []:
                    for row in table or []:
                        if not row:
                            continue
                        cleaned = [(cell or "").strip() for cell in row]
                        # keep only rows that have at least one non-empty cell
                        if any(cleaned):
                            table_parts.append(",".join(cleaned))
            except Exception:
                # we don't want the whole app to crash on weird tables
                pass

    # Prefer normal text; if almost nothing, fall back to tables
    text = "\n".join(text_parts)
    if len(text.strip()) < 50 and table_parts:
        st.warning("Low-text PDF detected → using table extraction instead.")
        text = "\n".join(table_parts)

    return text


# === AI EXTRACT CSV (OpenAI model) ===
def ai_parse_transactions(text: str) -> str | None:
    """
    Calls OpenAI to extract transactions as CSV with header:
    merchant,amt,trans_date_trans_time,description
    """
    prompt = f"""
Extract ALL bank transactions from this text into perfect CSV.
Output ONLY this exact header and data (no extra text):

merchant,amt,trans_date_trans_time,description

Rules:
- merchant = main payee (e.g., ZOMATO, UBER, SALARY, ATM)
- amt = number only, negative = expense, positive = income
- trans_date_trans_time = date or date+time
- description = full original line

Text:
{text[:28000]}

Return only CSV starting with header.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # or "gpt-4o-mini" / any chat-capable OpenAI model you prefer
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        # Safely extract the content from the first choice, handling None and different shapes
        choice = response.choices[0]
        content = None

        # Try attribute access for message.content
        msg = getattr(choice, "message", None)
        if msg is not None:
            if isinstance(msg, dict):
                content = msg.get("content")
            else:
                content = getattr(msg, "content", None)

        # Fallback to other possible fields
        if not content:
            try:
                content = getattr(choice, "text", None)
            except Exception:
                content = None

        # Only call strip() if content is a non-empty string
        if content and isinstance(content, str):
            csv = content.strip()
            if csv.startswith("merchant,amt"):
                return csv
    except Exception as e:
        st.error(f"AI error: {e}")
    return None

# === MAIN APP ===
uploaded_file = st.file_uploader("Upload Bank Statement PDF", type="pdf")

if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    path = "temp/statement.pdf"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

        # Step 1: Extract text
    with st.spinner("Reading PDF..."):
        text = extract_text_smart(path)
        st.info(f"Extracted {len(text):,} characters of raw text/tables.")

        # 🔍 optional debug: see what was extracted
        if st.checkbox("Show extracted text (debug)"):
            st.text_area("Raw extracted text (first 4000 chars)", text[:4000], height=250)

        # If still nothing, there's nothing useful to send to the LLM
        if len(text.strip()) == 0:
            st.error(
                "I couldn't extract any text or tables from this PDF.\n\n"
                "It is probably a fully scanned image or an unsupported template. "
                "For this demo, please upload a non-scanned, text-based bank statement."
            )
            st.stop()

        # Step 2: AI → CSV
    with st.spinner("AI is extracting transactions (OpenAI)..."):
        csv_content = ai_parse_transactions(text)
        if not csv_content:
            st.error("AI failed to parse. Please check the PDF or try again.")
            st.stop()
        
        df = pd.read_csv(StringIO(csv_content))

        # 🚨 IMPORTANT: guard against empty CSV (0 transactions)
        if df.empty:
            st.error(
                "The AI parser returned an empty CSV (0 transactions).\n\n"
                "This usually means it couldn't recognize any transaction lines "
                "in this PDF. Try another statement or tweak the PDF/prompt."
            )
            st.stop()

        st.success(f"Found {len(df)} transactions!")
        # Optional: preview a few rows
        st.dataframe(df.head(), use_container_width=True)


    # Step 3: Your model prediction
    with st.spinner("Categorizing with data_1.joblib..."):
        X = df.copy()

        # --- Rebuild the features expected by the saved model ---
        # text_col: text features (merchant + description)
        if "merchant" in X.columns:
            merchant_series = X["merchant"].fillna("")
        else:
            merchant_series = pd.Series("", index=X.index)

        if "description" in X.columns:
            description_series = X["description"].fillna("")
        else:
            description_series = pd.Series("", index=X.index)

        X["text_col"] = (merchant_series.astype(str) + " " + description_series.astype(str)).str.strip()

        # Amount numeric
        X["amt"] = pd.to_numeric(X["amt"], errors="coerce").fillna(0)

        # Parse datetime
        dt = pd.to_datetime(X["trans_date_trans_time"], errors="coerce")

        # day_of_week: 0 (Monday) – 6 (Sunday)
        X["day_of_week"] = dt.dt.weekday.fillna(0).astype(int)

        # hour: 0–23
        X["hour"] = dt.dt.hour.fillna(0).astype(int)

        # log_amount: simple log transform of absolute value (+1 for stability)
        X["log_amount"] = np.log1p(X["amt"].abs()) * np.sign(X["amt"])

        # amount_bin: rough bins on amount (adjust if you used something else in training)
        bins = [-10**9, -1000, -100, -10, 0, 10, 100, 1000, 10**9]
        X["amount_bin"] = pd.cut(
            X["amt"],
            bins=bins,
            labels=False,
            include_lowest=True
        ).fillna(0).astype(int)

        # Keep exactly the columns expected by the ColumnTransformer
        features = X[["hour", "log_amount", "amount_bin", "day_of_week", "text_col"]]
                # Just in case something went wrong and features is empty
        if features.empty:
            st.error(
                "No usable rows left after preprocessing (feature matrix is empty). "
                "Check that the parsed CSV actually contains valid data."
            )
            st.stop()

        # Predict
        predictions = model.predict(features)
        X["category"] = predictions

        # Spending / income for the dashboard (use original amt)
        X["spending"] = X["amt"].apply(lambda x: -x if x < 0 else 0)
        X["income"] = X["amt"].apply(lambda x: x if x > 0 else 0)

    # === DASHBOARD ===
    st.success("FINISHED!")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spending", f"₹{X['spending'].sum():,.0f}")
    c2.metric("Total Income", f"₹{X['income'].sum():,.0f}")
    c3.metric("Net", f"₹{X['income'].sum() - X['spending'].sum():,.0f}")

    # Pie chart
    spend_df = X[X["amt"] < 0].groupby("category")["spending"].sum().reset_index()
    if len(spend_df) > 0:
        fig = px.pie(spend_df, values="spending", names="category", title="Spending Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("Categorized Transactions")
    st.dataframe(
        X[["trans_date_trans_time", "merchant", "amt", "category"]].round(2),
        use_container_width=True,
    )

    # Download
    st.download_button(
        "Download Report",
        X.to_csv(index=False),
        "FinSentry_Report.csv",
        "text/csv",
    )