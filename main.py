import os
import json
import logging
import re
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# --- Configuration ---
class Settings(BaseSettings):
    groq_api_key: str
    classified_json_path: str = "transactions.json"
    model_name: str = "llama3-8b-8192"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

# --- Application State ---
class AppState:
    def __init__(self):
        self.settings: Optional[Settings] = None
        self.llm: Optional[ChatGroq] = None
        self.transactions: List[Dict] = []

app_state = AppState()

# --- Text Processing Utilities ---
def find_relevant_transactions(query: str, transactions: List[Dict], top_k: int = 5) -> List[Dict]:
    query = query.lower()
    relevant = []

    amount_match = re.search(r'(over|above|more than)\s*(\$|\u20b9|\u20ac|\u00a3)?(\d+)', query)
    amount_threshold = float(amount_match.group(3)) if amount_match else None

    for txn in transactions:
        score = 0
        if amount_threshold:
            try:
                if float(txn['amount']) > amount_threshold:
                    score += 2
            except (ValueError, KeyError):
                pass

        for field in ['party', 'category', 'type', 'date']:
            if field in txn and str(txn[field]).lower() in query:
                score += 1

        if score > 0:
            relevant.append((score, txn))

    relevant.sort(key=lambda x: x[0], reverse=True)
    return [txn for (score, txn) in relevant[:top_k]]

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = Settings()
        app_state.settings = settings

        with open(settings.classified_json_path, "r", encoding="utf-8") as f:
            app_state.transactions = json.load(f)
            if not isinstance(app_state.transactions, list):
                raise ValueError("Transaction data should be a list")
            logger.info(f"Loaded {len(app_state.transactions)} transactions")

        app_state.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.model_name,
            temperature=0.1
        )
        logger.info("LLM initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down")

# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)

# --- Models ---
class ChatRequest(BaseModel):
    user_input: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    relevant_transactions: Optional[List[Dict]] = None
    error: Optional[str] = None

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Transaction ChatBot API! Use /chat to interact."}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    try:
        if not app_state.llm or not app_state.transactions:
            raise HTTPException(status_code=503, detail="Service not ready")

        user_input = request.user_input.lower()

        if not any(keyword in user_input for keyword in [
            "spending", "transaction", "expense", "money", "category", "amount", "income", "credit", "debit"
        ]):
            generic_prompt = f"""You are a helpful assistant.\nA user says: \"{request.user_input}\"\nReply in a friendly and helpful tone."""
            response = app_state.llm.invoke(generic_prompt)
            return ChatResponse(response=response.content)

        relevant_txns = find_relevant_transactions(
            request.user_input,
            app_state.transactions,
            request.top_k
        )

        if not relevant_txns:
            transactions_text = "\n".join([
                f"{txn['date']}: {txn['type']} of {txn['amount']} to/from {txn.get('party', '?')} for {txn.get('category', '?')}"
                for txn in app_state.transactions[:15]
            ])
            context_info = f"No specific matches found. Here's a sample of recent transactions:\n{transactions_text}"
        else:
            context_info = "\n".join([
                f"{i+1}. {txn['type']} of {txn['amount']} with {txn.get('party', '?')} for {txn.get('category', '?')} on {txn.get('date', '?')}"
                for i, txn in enumerate(relevant_txns)
            ])

        financial_prompt = f"""
You are a financial assistant. Based on the transactions provided below, answer the user's question.

Transactions:
{context_info}

User Question: {request.user_input}

Include in your answer:
- Relevant transaction details
- Calculations if needed
- Observations about spending patterns
- Be precise with dates and amounts
"""
        response = app_state.llm.invoke(financial_prompt)
        return ChatResponse(
            response=response.content,
            relevant_transactions=relevant_txns
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return ChatResponse(
            response="Sorry, I encountered an error processing your request.",
            error=str(e)
        )

@app.get("/health")
async def health_check():
    return {
        "status": "ready" if app_state.llm and app_state.transactions else "initializing",
        "transactions_loaded": len(app_state.transactions) if app_state.transactions else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
