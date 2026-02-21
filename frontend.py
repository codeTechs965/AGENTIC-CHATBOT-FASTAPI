import streamlit as st
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="LangGraph Agent UI", layout="wide")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Agent Settings")

    system_prompt = st.text_area(
        "System Prompt",
        height=120,
        placeholder="Define your AI agent behavior..."
    )

    MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    MODEL_NAMES_GEMINI = ["gemini-flash-latest"]

    provider = st.radio("Select Provider:", ("Groq", "GEMINI"))

    if provider == "Groq":
        selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
    else:
        selected_model = st.selectbox("Select Gemini Model:", MODEL_NAMES_GEMINI)

    allow_web_search = st.checkbox("Allow Web Search")

    st.divider()
    st.caption("Backend URL: http://127.0.0.1:9999/chat")

# ---------------- MAIN CHAT AREA ----------------
st.title("🤖 AI Chatbot Agent")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Box (bottom style)
user_query = st.chat_input("Ask anything...")

API_URL = "http://127.0.0.1:9999/chat"

if user_query:
    # Show user message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare Payload
    payload = {
        "model_name": selected_model,
        "model_provider": provider,
        "system_prompt": system_prompt,
        "messages": user_query,
        "allow_search": allow_web_search
    }

    # Call Backend
    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            response_data = response.json()

            if isinstance(response_data, dict):
                assistant_reply = response_data.get("final_response", str(response_data))
            else:
                 assistant_reply = str(response_data)
        else:
            assistant_reply = f"Error: {response.status_code}"

    except Exception as e:
        assistant_reply = f"Connection Error: {str(e)}"

    # Show assistant reply
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_reply}
    )