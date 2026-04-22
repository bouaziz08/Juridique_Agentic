from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_core import RAGQueryProcessor

# Constants
CHROMA_PATH = "./chroma"  # database path
rag_processor = RAGQueryProcessor(chroma_path=CHROMA_PATH, model_name="mistral")

# FastAPI app instance
app = FastAPI()

# Pydantic model for request body
class QueryRequest(BaseModel):
    query_text: str

@app.post("/query")
def query_rag(request: QueryRequest):  # Accept the request as a Pydantic model
    try:
        # Extract query_text from the request
        result = rag_processor.process_query(request.query_text)
        print("result: ", result)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
