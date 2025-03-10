import streamlit as st
import pandas as pd
import os
import litellm
from langchain.chat_models import ChatOpenAI
from crewai import Agent, Task, Crew
import plotly.express as px
from io import BytesIO
from crewai_tools import CodeInterpreterTool

# API Setup
os.environ["OPENAI_API_KEY"] = "AIzaSyCSq35o-1vLYe3bKjKRoGNezTJNRmDMEx0"
litellm.api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name="gemini/gemini-1.5-flash", temperature=0.5)

# Streamlit App Title
st.title("AI-Powered CSV Analyzer 🚀")
st.markdown("Upload your CSV to clean, analyze, forecast, and visualize data.")

uploaded_file = st.file_uploader("📌 Upload CSV File", type=["csv"])

if uploaded_file:
    csv_data = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.dataframe(csv_data, height=300)
    csv_string = csv_data.to_csv(index=False)

    st.download_button("⬇️ Download Raw CSV", data=csv_string, file_name="raw_data.csv", mime="text/csv")

    # Agents
    data_cleaner = Agent(
        role="Data Cleaning Specialist", goal="Clean Data",
        backstory="Preprocessing expert.", verbose=True, llm=llm)

    clean_task = Task(
        description="Clean CSV: Remove nulls, fix typos, standardize dates.", agent=data_cleaner,
        expected_output="Cleaned CSV Text + Cleaning Report")

    eda_agent = Agent(role="EDA Specialist", goal="EDA Analysis", backstory="Data Insights Expert", verbose=True, llm=llm)
    eda_task = Task(description="Perform EDA and show stats.", agent=eda_agent, expected_output="EDA Summary", context=[clean_task])

    forecast_agent = Agent(role="Forecast Specialist", goal="Forecast Data", backstory="Time Series Expert", verbose=True, llm=llm)
    forecast_task = Task(description="Forecast next 7 days and merge with historical data.", agent=forecast_agent,
                         expected_output="Forecasted Table", context=[clean_task])

    viz_agent = Agent(role="Visualization Specialist", goal="Generate Visuals", backstory="Plotly Pro",
                      verbose=True, llm=llm)
    viz_task = Task(description="Generate charts from both historical and forecasted data.",
                    agent=viz_agent, expected_output="Visualizations", context=[clean_task, forecast_task])

    report_agent = Agent(role="Report Writer", goal="Generate Report", backstory="Documentation Expert",
                         verbose=True, llm=llm)
    report_task = Task(description="Generate Report with Cleaning, EDA, Forecast, and Visualizations.",
                       agent=report_agent, expected_output="PDF Report", context=[clean_task, eda_task, forecast_task, viz_task])

    crew = Crew(agents=[data_cleaner, eda_agent, forecast_agent, viz_agent, report_agent],
                tasks=[clean_task, eda_task, forecast_task, viz_task, report_task])

    if st.button("🔥 Run Analysis"):
        with st.spinner("⏳ Processing..."):
            try:
                result = crew.kickoff(inputs={"csv": csv_string})

                st.success("✅ Analysis Complete")

                # Cleaning Report
                st.markdown("### Data Cleaning Report")
                st.text(clean_task.output.raw.strip("'''markdown"))

                # Download Cleaned CSV
                cleaned_csv = clean_task.output.raw
                st.download_button("⬇️ Download Cleaned CSV", cleaned_csv, "cleaned_data.csv", mime="text/csv")

                # EDA Summary
                st.markdown("### EDA Summary")
                st.markdown(eda_task.output.raw.strip("'''markdown"))

                # Forecast Table
                st.markdown("### Forecast Preview")
                forecast_data = pd.read_csv(BytesIO(forecast_task.output.raw.encode()))  # Convert CSV text back to DataFrame
                st.dataframe(forecast_data)

                # Visualization Before & After Forecast
                st.markdown("### Visualizations")

                columns = csv_data.columns.tolist()
                selected_col = st.selectbox("Select Column to Visualize", columns)

                # Before Forecast
                st.markdown("**Before Forecast**")
                fig_before = px.line(csv_data, x=csv_data.index, y=selected_col, title=f"{selected_col} (Historical)")
                st.plotly_chart(fig_before)

                # After Forecast
                st.markdown("**After Forecast**")
                fig_after = px.line(forecast_data, x=forecast_data.index, y=selected_col, title=f"{selected_col} (Forecast Included)")
                st.plotly_chart(fig_after)

                # Download Report
                st.download_button("📄 Download Report", report_task.output.raw, "final_report.pdf", mime="application/pdf")

            except Exception as e:
                st.error(f"❌ Error: {e}")

