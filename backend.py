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

ALLOWED_MODEL_NAMES=["mixtral-8x7b-32768","llama3-70b-8192","llama-3.3-70b-versatile","gemini-flash-latest"]
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
