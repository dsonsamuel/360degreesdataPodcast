import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from PIL import Image

# -------------------------------
# ⚙️ Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Analyst - Powered by PandasAI + GPT")
st.markdown("""
Ask questions about your dataset in **plain English** and get answers with **charts and insights**.  
Upload a CSV file below to begin.
""")

# -------------------------------
# 🧠 Sidebar Setup
# -------------------------------
st.sidebar.header("🔑 API Configuration")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# -------------------------------
# 📂 File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write("### 🧾 Preview of your data:")
    st.dataframe(df.head())

    # -------------------------------
    # 🤖 Initialize LLM + PandasAI
    # -------------------------------
    llm = OpenAI(api_token=openai_api_key)
    smart_df = SmartDataframe(df, config={"llm": llm, "enable_cache": True})

    # -------------------------------
    # 💬 Chat Interface with Memory
    # -------------------------------
    st.divider()
    st.subheader("💬 Chat with your data")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask a question about your dataset:")
    
    if user_query:
        with st.spinner("Thinking..."):
            try:
                result = smart_df.chat(user_query)

                if isinstance(result, str) and result.endswith((".png", ".jpg")) and os.path.exists(result):
                    st.write("Here's the chart generated from your query:")
                    image = Image.open(result)
                    st.image(image, caption="Generated Visualization", use_container_width=True)


                # Display response
                st.markdown(f"**Answer:** {result}")
                st.session_state.chat_history.append((user_query, result))

                # Check if visualization exists in PandasAI temp folder
                img_path = os.path.join(os.getcwd(), "exports", "temp_chart.png")
                if os.path.exists(img_path):
                    st.image(img_path, caption="Generated Visualization", use_container_width=True)
                    os.remove(img_path)

            except Exception as e:
                st.error(f"Error: {e}")

    # -------------------------------
    # 🕒 Show Chat History
    # -------------------------------
    if st.session_state.chat_history:
        st.divider()
        st.subheader("🗂 Chat History")
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {a}")
            st.markdown("---")

else:
    st.info("👆 Please upload a CSV file to start exploring your data.")


