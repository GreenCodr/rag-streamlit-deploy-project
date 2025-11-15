import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="RAG SQL Assistant", layout="wide")
st.title("🧠 RAG SQL Assistant (Instacart Database)")
st.write("Ask any question about the Instacart dataset. The backend will process it using RAG + SQL logic.")

API_URL = "http://127.0.0.1:8000/query"

# User input section
question = st.text_input("💬 Enter your question here:")
mode = st.selectbox("Select mode:", ["auto", "retrieval", "aggregation"])
top_k = st.slider("Top K results", 3, 15, 6)

if st.button("Submit"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your query..."):
            payload = {
                "question": question,
                "mode": mode,
                "top_k": top_k
            }
            try:
                response = requests.post(API_URL, json=payload)
                result = response.json()

                st.subheader("📝 Answer")
                st.write(result.get("answer_text", ""))

                # Display data if any
                if "data" in result and isinstance(result["data"], list):
                    df = pd.DataFrame(result["data"])
                    st.subheader("📄 Retrieved Data")
                    st.dataframe(df)

                # Generate chart if chart_type is provided
                chart_type = result.get("chart_type", "none")
                chart_spec = result.get("chart_spec", {})

                if chart_type != "none" and "data" in result:
                    try:
                        df_chart = pd.DataFrame(result["data"])
                        st.subheader("📊 Chart")

                        if chart_type == "bar":
                            fig = px.bar(df_chart, x="x", y="y", title="Bar Chart")
                        elif chart_type == "line":
                            fig = px.line(df_chart, x="x", y="y", title="Line Chart")
                        elif chart_type == "pie":
                            fig = px.pie(df_chart, names="x", values="y", title="Pie Chart")
                        else:
                            fig = None

                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as chart_error:
                        st.error(f"Chart error: {chart_error}")

            except Exception as e:
                st.error(f"Error contacting backend: {e}")