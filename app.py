"""
app.py
------
Streamlit UI — the single entry point for the DataLens app.

Flow:
    1. User uploads a CSV.
    2. We show a quick preview (head + shape).
    3. User clicks "Generate Insights".
    4. We call engine.generate_insights(df) and render each finding with a
       colored severity icon.
    5. We also show a "Column Overview" table built from profiling.py, so
       the profiling module gets reused instead of duplicating logic.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from profiling import profile_columns
from engine import generate_insights

# Maps each severity level to a colored icon shown next to its message.
SEVERITY_ICONS = {
    "critical": "🔴",
    "warning": "🟠",
    "info": "🔵",
}

st.set_page_config(page_title="DataLens", page_icon="📊", layout="centered")

st.title("📊 DataLens")
st.caption(
    "Upload a CSV and get an automated, rule-based profile of your data "
    "— missing values, skew, correlation, outliers and more."
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # --- Load the CSV, with basic error handling ------------------------
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error("This file appears to be empty. Please upload a valid CSV with data.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Couldn't parse this file as CSV. Please check the file format.")
        st.stop()
    except Exception as e:
        st.error(f"Something went wrong while reading the file: {e}")
        st.stop()

    if df.shape[0] == 0 or df.shape[1] == 0:
        st.warning("The uploaded CSV has no rows or no columns to analyze.")
        st.stop()

    # --- Preview ----------------------------------------------------------
    st.subheader("Preview")
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head())

    # --- Generate Insights button -----------------------------------------
    st.subheader("Insights")
    if st.button("Generate Insights", type="primary"):
        with st.spinner("Analyzing your data..."):
            try:
                findings = generate_insights(df)
            except Exception as e:
                st.error(f"Something went wrong while generating insights: {e}")
                findings = None

        if findings is not None:
            if len(findings) == 0:
                st.success("✅ No significant issues detected in this dataset.")
            else:
                for finding in findings:
                    icon = SEVERITY_ICONS.get(finding["severity"], "⚪")
                    st.markdown(f"{icon} **{finding['column']}** — {finding['message']}")

    # --- Column Overview table --------------------------------------------
    st.subheader("Column Overview")
    try:
        profile = profile_columns(df)
        overview_rows = []
        for col, info in profile.items():
            overview_rows.append({
                "Column": col,
                "Type": info["type"],
                "Dtype": info["dtype"],
                "Missing %": info["missing_pct"],
                "Unique Values": info["unique_count"],
                "Skew": info["skew"] if info["skew"] is not None else "-",
                "Std Dev": info["std"] if info["std"] is not None else "-",
            })
        overview_df = pd.DataFrame(overview_rows)
        st.dataframe(overview_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not build column overview: {e}")

    # --- Correlation heatmap (only if there are 2+ numeric columns) -------
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        st.subheader("Correlation Heatmap")
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        finally:
            # Always close the figure, even if rendering fails — matplotlib
            # keeps every unclosed figure in memory, and on Streamlit's app
            # reruns (which happen on every interaction) this leaks memory
            # and can eventually crash the app, especially on the free
            # hosting tier's limited RAM.
            plt.close(fig)

else:
    st.info("👆 Upload a CSV file to get started.")
