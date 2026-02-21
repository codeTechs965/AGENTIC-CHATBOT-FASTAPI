#agent

# Setup API for GROK OPEN AI AND TAVILY
# SETUP LLM AND TOOL
# SETUP AI AGENT WIT SEARCH TOOL FUNCTIONALITY    

# STEP 1
from dotenv import load_dotenv
load_dotenv()
import os
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")
 

# STEP2

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

gemini_llm=ChatGoogleGenerativeAI(model="gemini-flash-latest")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

   

#step3
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

def get_response_from_ai_agent(llm_id,query,allow_search, system_prompt, provider):
    if not isinstance(query, str):
        query = str(query)

    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="GEMINI":
        llm=ChatGoogleGenerativeAI(model=llm_id)
   
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
