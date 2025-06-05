from duckduckgo_search import DDGS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from typing import *
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pydantic import BaseModel
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not HUGGINGFACE_TOKEN or not GOOGLE_API_KEY:
    logger.warning("Environment variables not fully set. Using fallback values for testing.")
    HUGGINGFACE_TOKEN = HUGGINGFACE_TOKEN or "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"
    GOOGLE_API_KEY = GOOGLE_API_KEY or "AIzaSyBVusfvWlCSmOKbw6KUQPBl9IYqvEDZnOk"

# Global variables for lazy loading
tokenizer = None
bert_model = None
search = None
llm = None
chain = None
embedding_model = None
models_loaded = False
executor = ThreadPoolExecutor(max_workers=2)

async def load_models_async():
    """Async lazy load models to reduce startup time"""
    global tokenizer, bert_model, search, llm, chain, embedding_model, models_loaded
    
    if models_loaded:
        return
    
    try:
        logger.info("Starting model loading...")
        start_time = time.time()
        
        # Load search first (fastest)
        search = DDGS()
        logger.info("✓ Search initialized")
        
        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("✓ Embedding model loaded")
        
        # Load BERT components
        tokenizer = BertTokenizer.from_pretrained("mohamedzabady/bert-fake-news", token=HUGGINGFACE_TOKEN)
        logger.info("✓ BERT tokenizer loaded")
        
        bert_model = BertForSequenceClassification.from_pretrained("mohamedzabady/bert-fake-news", token=HUGGINGFACE_TOKEN)
        device = torch.device("cpu")  # Force CPU for Railway
        bert_model.to(device)
        bert_model.eval()
        logger.info("✓ BERT model loaded")
        
        # Load LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.0,
        )
        
        system = """
        You are a concise fact-checking assistant. Analyze news reports and determine if claims are REAL or FAKE.

        ## Output Format (MANDATORY):
        Start with: **REAL (X%)** or **FAKE (X%)** where X is your confidence percentage.

        ## Response Structure:
        **REAL/FAKE (X%)**

        **Reasoning:** Brief explanation based on trusted sources.

        **Sources:** Number of trusted vs untrusted sources.

        ## Guidelines:
        - **REAL**: Supported by trusted sources
        - **FAKE**: Contradicted by trusted sources OR only untrusted sources support it
        - **Confidence**: 60-75% = Some evidence, 76-89% = Strong evidence, 90%+ = Very strong evidence
        - **Keep responses short** - 2-3 sentences maximum for reasoning
        - **Always use markdown formatting**
        - **Prioritize trusted sources heavily**

        ## Example:
        **FAKE (85%)**

        **Reasoning:** Multiple trusted news sources contradict this claim with verified information.

        **Sources:** 3 trusted sources against, 1 untrusted source supporting.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{input}"),
        ])
        chain = prompt | llm
        logger.info("✓ LLM chain created")
        
        models_loaded = True
        load_time = time.time() - start_time
        logger.info(f"🎉 All models loaded successfully in {load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise e

class NewsRequest(BaseModel):
    input_str: str

def check_relevance(query: str):
    """Check relevance of news articles to query"""
    try:
        news_list = search.news(query, max_results=5)  # Limit for performance
        relevant_news = [] 
        for news in news_list:
            if news.get('title') and news.get('source'):
                cosine_similarity_score = cosine_similarity(
                    [embedding_model.embed_query(query)], 
                    [embedding_model.embed_query(news["title"])]
                )
                if cosine_similarity_score > 0.3:
                    relevant_news.append(news)
        return relevant_news
    except Exception as e:
        logger.error(f"Error in check_relevance: {str(e)}")
        return []

def check_trusted(news: dict):
    """Check if news source is trusted using BERT model"""
    try:
        source = news.get('source', 'unknown')
        inputs = tokenizer(source, return_tensors="pt", padding=True, truncation=True, max_length=128)
        device = torch.device("cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            is_trusted = bool(prediction)

        return is_trusted
    except Exception as e:
        logger.error(f"Error in check_trusted: {str(e)}")
        return False

async def generate_content(query: str):
    """Generate fact-check content"""
    # Ensure models are loaded
    if not models_loaded:
        await load_models_async()
    
    try:
        trusted_template = ""
        untrusted_template = ""
        
        # Run relevance check in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, check_relevance, query)
        
        if not results:
            messages = f"""
                No relevant news was found for: '{query}'. Possible reasons:
                - The topic is too new or niche.
                - The query may not be news-related.
                - Limited data coverage in our system.
                
                Suggestions:
                1. Rephrase your search.
                2. Check real-time sources """

            response = await loop.run_in_executor(executor, lambda: chain.invoke({"input": messages}))
            return response.content
        
        # Process results
        trusted_count = 0
        untrusted_count = 0
        
        for news in results:
            is_trusted = await loop.run_in_executor(executor, check_trusted, news)
            
            if is_trusted:
                trusted_count += 1
                trusted_template += f"""
        **Title:** {news.get('title', 'N/A')}  
        **Source:** {news.get('source', 'N/A')} (Trusted)  
        **Date:** {news.get('date', 'N/A')}  
        **Summary:** {news.get('body', 'N/A')[:200]}...
        **Read More:** [Full Article]({news.get('url', '#')})\n\n
                """
            else:
                untrusted_count += 1
                untrusted_template += f"""
        ⚠️ Source not trusted. These news might be fake
        **Source:** {news.get('source', 'N/A')} (Untrusted)\n
                """
        
        # Generate final response
        final_query = (
            trusted_template + untrusted_template + 
            f"\n\nTrusted sources: {trusted_count}, Untrusted sources: {untrusted_count}\n"
            f"Based on these reports, is the query \"{query}\" TRUE or FALSE and WHY?\n\n"
            "MARKDOWN Answer:"
        )
        
        response = await loop.run_in_executor(executor, lambda: chain.invoke({"input": final_query}))
        return response.content
        
    except Exception as e:
        logger.error(f"Error in generate_content: {str(e)}")
        return f"**ERROR**\n\nUnable to process the query due to: {str(e)}"

# FastAPI app
app = FastAPI(
    title="Fact Checker API",
    version="1.0.0",
    description="AI-powered fact checking API using multiple ML models"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rashed-five.vercel.app",
        "https://rashed-five.vercel.app/",
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.railway.app",  # Railway domains
        "*"  # Allow all for testing (secure this in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.on_event("startup")
async def startup_event():
    """Warm up models on startup"""
    logger.info("🚀 Starting Fact Checker API...")
    # Don't load models on startup to reduce cold start time
    # They'll be loaded on first request

@app.get("/")
async def root():
    return {
        "message": "🔍 Fact Checker API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": models_loaded,
        "endpoints": {
            "health": "/health",
            "fact_check": "/Trustness/",
            "warmup": "/warmup"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "timestamp": time.time()
    }

@app.post("/warmup")
async def warmup():
    """Endpoint to warm up the models"""
    try:
        await load_models_async()
        return {
            "message": "✅ Models warmed up successfully!",
            "models_loaded": models_loaded
        }
    except Exception as e:
        return {
            "message": f"❌ Error warming up models: {str(e)}",
            "models_loaded": models_loaded
        }

@app.post("/Trustness/")
async def check_trustness(request: NewsRequest):
    """Main fact-checking endpoint"""
    try:
        logger.info(f"Processing fact-check request for: {request.input_str[:50]}...")
        
        start_time = time.time()
        trusted_text = await generate_content(request.input_str)
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Request processed in {processing_time:.2f} seconds")
        
        return {
            "trusted_text": trusted_text,
            "query": request.input_str,
            "processing_time": processing_time,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"❌ Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Processing failed",
                "message": str(e),
                "query": request.input_str
            }
        )

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)