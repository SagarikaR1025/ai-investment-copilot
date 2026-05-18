import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath("src"))

from rag_pipeline import query_research_notes
from agents import risk_analysis_agent, investment_strategy_agent, executive_summary_agent, final_recommendation_agent

st.set_page_config(
    page_title="AI Investment Research Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI Investment Research Copilot")

st.markdown("""
    Analyze market trends, trading activity, and AI-generated investment insights
    using financial data pipelines and intelligent analytics workflows.
    """)

st.markdown(
    """
    ### AI Workflow

    This platform combines:
    - Financial market data ingestion
    - Retrieval-Augmented Generation (RAG)
    - Vector database semantic search
    - Multi-agent AI analysis workflows
    - Executive investment recommendation generation
    """
)

st.divider()

df = pd.read_csv("data/raw/all_stock_data.csv")

st.sidebar.header("Filters")
st.sidebar.markdown(
    "Select a company ticker to explore market trends and investment signals."
)
company_descriptions = {
    "AAPL": "Apple — Consumer technology and services ecosystem",
    "NVDA": "NVIDIA — AI infrastructure and semiconductor leader",
    "TSLA": "Tesla — Electric vehicles, energy, and autonomy"
}

selected_ticker = st.sidebar.selectbox(
    "Select Company",
    sorted(df["Ticker"].unique())
)

st.sidebar.info(company_descriptions[selected_ticker])

filtered_df = df[df["Ticker"] == selected_ticker].copy()

st.header("Market Performance Overview")

latest_close = round(filtered_df["Close"].iloc[-1], 2)
highest_close = round(filtered_df["Close"].max(), 2)
lowest_close = round(filtered_df["Close"].min(), 2)
avg_volume = round(filtered_df["Volume"].mean(), 0)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Latest Close", f"${latest_close}")
col2.metric("6M High", f"${highest_close}")
col3.metric("6M Low", f"${lowest_close}")
col4.metric("Avg Volume", f"{avg_volume:,.0f}")

st.divider()

st.header("Stock Price Trend")

price_fig = px.line(
    filtered_df,
    x="Date",
    y="Close",
    title=f"{selected_ticker} Closing Price Over Time",
    markers=True
)
price_fig.update_layout(xaxis_title="Date", yaxis_title="Closing Price (USD)")

st.plotly_chart(price_fig, use_container_width=True)

st.header("Trading Volume")

volume_fig = px.bar(
    filtered_df,
    x="Date",
    y="Volume",
    title=f"{selected_ticker} Trading Volume Activity"
)

volume_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Volume"
)

st.plotly_chart(volume_fig, use_container_width=True)

st.divider()

st.header("AI-Powered Investment Analysis")

st.info(
    f"""
    Initial research summary for {selected_ticker}:

    • Recent price movement and volume patterns have been loaded  
    • Risk, sentiment, and earnings transcript analysis will be added next  
    • Upcoming phases will include RAG retrieval, AI agents, and automated investment research summaries  
    """
)

st.divider()

st.header("Underlying Market Data")

st.dataframe(
    filtered_df.tail(20),
    use_container_width=True
)
st.divider()

st.header("AI Research Assistant")

user_question = st.text_input(
    "Ask a financial research question:"
)

if user_question:

    results = query_research_notes(user_question)

    retrieved_docs = results["documents"][0]
    combined_context = "\n\n".join(retrieved_docs)

    st.header("Executive Research Summary")

    with st.spinner("Generating executive summary..."):
        executive_summary = executive_summary_agent(
            combined_context,
            selected_ticker
        )

    st.write(executive_summary)

    st.divider()

    st.subheader("Retrieved Research Context")

    for idx, doc in enumerate(retrieved_docs):

        with st.expander(f"Research Match {idx + 1}"):

            st.write(doc)

    st.header("Claude Risk Analysis Agent")

    with st.spinner("Generating risk analysis..."):
        risk_summary = risk_analysis_agent(
            combined_context,
            selected_ticker
        )

    st.warning(risk_summary)

    st.divider()

    st.header("Claude Investment Strategy Agent")

    with st.spinner("Generating investment outlook..."):
        strategy_summary = investment_strategy_agent(
            combined_context,
            selected_ticker
        )

    st.info(strategy_summary)
    st.divider()

    st.header("Final AI Investment Recommendation")

    with st.spinner("Generating final recommendation..."):
        final_recommendation = final_recommendation_agent(
            combined_context,
            selected_ticker
        )

    st.success(final_recommendation)

    st.divider()

st.caption(
    "Built using Streamlit, ChromaDB, Claude API, semantic retrieval, and agentic AI workflows."
)