#agent

# Setup API for GROK OPEN AI AND TAVILY
# SETUP LLM AND TOOL
# SETUP AI AGENT WIT SEARCH TOOL FUNCTIONALITY    

# STEP 1
from dotenv import load_dotenv
load_dotenv()
import os
# GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
# TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
# OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
 

# STEP2

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

openai_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

   

#step3
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

def get_response_from_ai_agent(llm_id,query,allow_search, system_prompt, provider):
    if not isinstance(query, str):
        query = str(query)

    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)
   
    tools = [TavilySearch(max_results=2)] if allow_search else []       
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    ai_message = response["messages"][-1].content

    return (ai_message)
-------------------------------------------------------------------------------------------------------
#backend

# setup pydentic model(schema validation)
from pydantic import BaseModel
class RequestState(BaseModel):
    model_name : str
    model_provider :str
    system_prompt :str
    messages:str
    allow_search :bool


# setup ai agent from front end request
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

ALLOWED_MODEL_NAMES=["mixtral-8x7b-32768","llama3-70b-8192","llama-3.3-70b-versatile","gpt-4o-mini"]
app =FastAPI(title="Lang graph AI-Agent")

@app.post("/chat")
def chat_endpoints(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}
    #create AI Agent
    llm_id=request.model_name
    query=request.messages
    allow_search=request.allow_search
    system_prompt=request.system_prompt
    provider=request.model_provider
    response=get_response_from_ai_agent(llm_id,query,allow_search, system_prompt, provider)
    return response
    
# run app and explore swagger UI DOCS
if __name__== "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
    
-----------------------------------------------------------------------------------------------------------

#ui interface

# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()


#Step1: Setup UI with streamlit (model provider, model, system prompt, web_search, query)
import streamlit as st

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

system_prompt=st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider=st.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search=st.checkbox("Allow Web Search")

user_query=st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

API_URL="http://127.0.0.1:9999/chat"

if st.button("Ask Agent!"):
    if user_query.strip():
        #Step2: Connect with backend via URL
        import requests

        payload={
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": user_query,
            "allow_search": allow_web_search
        }

        response=requests.post(API_URL, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response_data}")

-----------------------------------------------------------------------------------------------------------
#env

EMAIL_ADDRESS=talhaammar57@gmail.com
EMAIL_PASSWORD=iccs sslm psgx wihd

GROQ_API_KEY=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
TAVILY_API_KEY=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
OPENAI_API_KEY=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


