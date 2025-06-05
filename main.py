from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "test")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "test")

app = FastAPI(title="Fact Checker API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    input_str: str

@app.get("/")
async def root():
    return {
        "message": "🔍 Fact Checker API is running!",
        "status": "healthy",
        "version": "1.0.0 (minimal)",
        "env_check": {
            "huggingface_token": "✅ Set" if HUGGINGFACE_TOKEN != "test" else "❌ Missing",
            "google_api_key": "✅ Set" if GOOGLE_API_KEY != "test" else "❌ Missing"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/Trustness/")
async def check_trustness(request: NewsRequest):
    """Minimal fact-checking endpoint for testing"""
    try:
        # For now, return a simple response to test deployment
        return {
            "trusted_text": f"**TESTING (50%)**\n\n**Reasoning:** This is a test response for the query: '{request.input_str}'. The full ML models will be loaded once deployment is successful.\n\n**Sources:** Testing phase - no sources checked yet.",
            "query": request.input_str,
            "status": "test_mode"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
