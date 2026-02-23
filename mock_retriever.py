from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/retrieve")
async def retrieve(payload: dict):
    dummy_doc = {"document": {"contents": "No Title\nNo content available."}}
    return {"result": [[dummy_doc] * payload.get("topk", 3)]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)