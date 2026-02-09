#!/usr/bin/env python3
"""
LangChain-based RAG query system for ZebraLens
Supports both local LLM and Google Gemini 3 Pro
"""

import os
import json
import argparse
import requests
from typing import List, Dict, Any, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Google Gemini (lightweight)
import google.generativeai as genai

# Configuration - Local LLM
INDEX_DIR = "vectorstore"
EMBED_MODEL = "all-MiniLM-L6-v2"
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"
TIMEOUT = 60

# Configuration - Gemini
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# System prompt for all LLM calls (English)
SYSTEM_PROMPT = (
    "You are ZebraLens, a clinical retrieval assistant. Provide a clear, narrative comparison between the user's query and the retrieved clinical case."
    " Write in simple paragraphs without tables, bullet points, or complex formatting."
    " Focus on the key similarities and differences in clinical presentation, demographics, and outcomes."
    " Only use information explicitly stated in the provided case text. Reference the case ID when mentioning specific findings."
    " Keep your response concise (2-3 paragraphs maximum) and readable. Do NOT provide medical advice or diagnoses."
)

# Global cache for heavy models
_vectorstore_cache = None
_embedding_model_cache = None


def get_embedding_model():
    """Lazy load and cache the SentenceTransformer model"""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        print("Lazy loading SentenceTransformer model...")
        import numpy as np
        from sentence_transformers import SentenceTransformer
        _embedding_model_cache = SentenceTransformer(EMBED_MODEL)
    return _embedding_model_cache


def load_vectorstore():
    """Load the pre-built vectorstore (Lazy loaded)"""
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache
        
    if not os.path.exists(INDEX_DIR):
        print(f"Warning: Vectorstore directory {INDEX_DIR} not found.")
        return None
    
    print("Lazy loading FAISS vectorstore...")
    # Lazy imports to save memory/time on startup
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    _vectorstore_cache = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return _vectorstore_cache


# Re-implement format_documents locally to avoid importing Document
def format_documents(docs) -> List[Dict[str, Any]]:
    """Format LangChain documents for display"""
    cases = []
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata
        case = {
            "id": metadata.get("id", "Unknown"),
            "title": metadata.get("title", "No title"),
            "content": content,
            "metadata": metadata
        }
        cases.append(case)
    return cases


def compute_similarity_percent(query: str, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute cosine similarity between query and each case content"""
    if not cases:
        return cases

    import numpy as np
    
    model = get_embedding_model()
    
    # encode query and documents
    q_emb = model.encode([query], convert_to_numpy=True)
    doc_texts = [c['content'] for c in cases]
    doc_embs = model.encode(doc_texts, convert_to_numpy=True)

    # normalize
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

    sims = (doc_embs @ q_emb.T).squeeze()
    # map to percent
    percents = ((sims + 1.0) / 2.0) * 100.0

    # If single result, sims is 0-d array
    if not isinstance(sims, np.ndarray) or sims.ndim == 0:
        sims = [float(sims)]
        percents = [float(percents)]

    for c, sim, p in zip(cases, sims, percents):
        c['score_raw'] = float(sim)
        c['score_percent'] = round(float(p), 2)

    return cases


# --- LLM Functions (Lightweight) ---

class LocalLLM:
    """Custom LangChain LLM wrapper for local OpenAI-compatible API"""
    def __init__(self):
        self.url = API_URL
        
    def __call__(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        # Truncate prompt
        max_prompt_length = 4000
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "\n\n[TRUNCATED]"
        
        body = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1,
            "stream": False
        }
        try:
            response = requests.post(self.url, headers=headers, json=body, timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            return "LLM Error: Empty response from API"
        except Exception as e:
            return f"LLM Error: {str(e)}"

def test_llm_connection() -> tuple[bool, str]:
    """Test connection to local LLM"""
    llm = LocalLLM()
    try:
        response = llm("Hello")
        return True, response.strip()
    except Exception as e:
        return False, str(e)

def call_llm_messages_gemini(messages: List[Dict[str, str]], timeout: int = TIMEOUT) -> str:
    """Call Google Gemini API"""
    if not GEMINI_API_KEY:
        return "Gemini Error: GEMINI_API_KEY not set"
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        full_prompt = f"{system_content}\n\n{user_content}" if system_content else user_content
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=1000)
        )
        return response.text if response.text else "Gemini Error: Empty response"
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def call_llm_messages_local(messages: List[Dict[str, str]], timeout: int = TIMEOUT) -> str:
    """Call local OpenAI-compatible API"""
    headers = {"Content-Type": "application/json"}
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        r = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
        r.raise_for_status()
        res = r.json()
        if isinstance(res, dict) and res.get("choices"):
            return res["choices"][0]["message"]["content"]
        return "LLM Error: empty or unexpected response"
    except Exception as e:
        return f"LLM Error: {str(e)}"

def call_llm_messages(messages: List[Dict[str, str]], timeout: int = TIMEOUT, use_gemini: bool = True) -> str:
    if use_gemini:
        return call_llm_messages_gemini(messages, timeout)
    else:
        return call_llm_messages_local(messages, timeout)

if __name__ == "__main__":
    # If run as script, load everything for CLI usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    
    vs = load_vectorstore()
    if vs:
        docs = vs.as_retriever(search_kwargs={"k": args.k}).get_relevant_documents(args.query)
        cases = format_documents(docs)
        cases = compute_similarity_percent(args.query, cases)
        print(f"Found {len(cases)} cases")
