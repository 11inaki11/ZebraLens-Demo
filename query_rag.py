#!/usr/bin/env python3
"""
LangChain-based RAG query system for ZebraLens
Supports both local LLM and Google Gemini 3 Pro
"""

import os
import json
import argparse
import requests
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Google Gemini
import google.generativeai as genai

# Configuration - Local LLM
INDEX_DIR = "vectorstore"
EMBED_MODEL = "all-MiniLM-L6-v2"
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"
TIMEOUT = 60

# Configuration - Gemini
GEMINI_MODEL = "gemini-3-flash-preview"
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


class LocalLLM(LLM):
    """Custom LangChain LLM wrapper for local OpenAI-compatible API"""
    
    @property
    def _llm_type(self) -> str:
        return "local_openai"
    
    def _call(
        self,
        prompt: str,
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs: Any,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        
        # Truncate prompt if too long to avoid 400 errors
        max_prompt_length = 4000  # Conservative limit
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
            response = requests.post(API_URL, headers=headers, json=body, timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "LLM Error: Empty response from API"
                
        except requests.exceptions.HTTPError as e:
            return f"LLM HTTP Error: {e.response.status_code} - {e.response.text[:200]}..."
        except requests.exceptions.Timeout:
            return "LLM Error: Request timeout"
        except requests.exceptions.ConnectionError:
            return "LLM Error: Cannot connect to local LLM server"
        except Exception as e:
            return f"LLM Error: {str(e)}"


def test_llm_connection() -> tuple[bool, str]:
    """Test connection to local LLM"""
    llm = LocalLLM()
    try:
        response = llm("Hello, respond with 'Connection OK'")
        return True, response.strip()
    except Exception as e:
        return False, str(e)


def call_llm_messages_gemini(messages: List[Dict[str, str]], timeout: int = TIMEOUT) -> str:
    """Call Google Gemini API with a messages list and return content or error string."""
    if not GEMINI_API_KEY:
        return "Gemini Error: GEMINI_API_KEY not set in .env file"
    
    try:
        # Convert OpenAI-style messages to Gemini format
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Extract system prompt and combine with user message
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        
        # Combine system and user content for Gemini
        full_prompt = f"{system_content}\n\n{user_content}" if system_content else user_content
        
        # Generate response
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=1000,
            )
        )
        
        if response.text:
            return response.text
        return "Gemini Error: Empty response"
        
    except Exception as e:
        return f"Gemini Error: {str(e)}"


def call_llm_messages_local(messages: List[Dict[str, str]], timeout: int = TIMEOUT) -> str:
    """Call local OpenAI-compatible API with a messages list and return content or error string."""
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
    except requests.exceptions.HTTPError as e:
        return f"LLM HTTP Error: {e.response.status_code} - {e.response.text[:200]}..."
    except requests.exceptions.Timeout:
        return "LLM Error: Request timeout"
    except requests.exceptions.ConnectionError:
        return "LLM Error: Cannot connect to local LLM server"
    except Exception as e:
        return f"LLM Error: {str(e)}"


def call_llm_messages(messages: List[Dict[str, str]], timeout: int = TIMEOUT, use_gemini: bool = True) -> str:
    """Route LLM call to either Gemini or local backend based on use_gemini flag.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        timeout: Request timeout in seconds
        use_gemini: If True, use Gemini API; if False, use local LLM
    
    Returns:
        LLM response text or error string
    """
    if use_gemini:
        return call_llm_messages_gemini(messages, timeout)
    else:
        return call_llm_messages_local(messages, timeout)


def load_vectorstore() -> FAISS:
    """Load the pre-built vectorstore"""
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"Vectorstore directory {INDEX_DIR} not found. Run build_index.py first.")
    
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def format_documents(docs: List[Document]) -> List[Dict[str, Any]]:
    """Format LangChain documents for display"""
    cases = []
    
    for doc in docs:
        # Extract content and metadata
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
    """Compute cosine similarity between query and each case content and add score_percent field.

    We map cosine similarity (-1..1) to a 0..100 percent scale via (sim+1)/2*100 so it's easier to read.
    """
    if not cases:
        return cases

    model = SentenceTransformer(EMBED_MODEL)
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

    for c, sim, p in zip(cases, sims, percents):
        c['score_raw'] = float(sim)
        c['score_percent'] = round(float(p), 2)

    return cases


def print_header(query: str, k: int):
    """Print formatted header"""
    print("=" * 80)
    print("ZEBRA LENS - LANGCHAIN RAG SYSTEM")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Retrieving top {k} cases...")


def print_connection_test():
    """Test and print LLM connection status"""
    print("\n0) Testing LLM connection...")
    connection_ok, test_response = test_llm_connection()
    if connection_ok:
        print(f"✓ LLM connection successful: {test_response}")
    else:
        print(f"✗ LLM connection failed: {test_response}")
    return connection_ok


def print_raw_cases(cases: List[Dict[str, Any]]):
    """Print raw case data from database"""
    print("\n" + "=" * 80)
    print("1) RAW DATABASE EXTRACTS")
    print("=" * 80)
    for i, case in enumerate(cases, 1):
        print(f"\n--- CASE {i} [ID: {case['id']}] ---")
        print(f"Title: {case['title']}")
        # show similarity metrics if available
        if 'score_raw' in case and 'score_percent' in case:
            print(f"Similarity: {case['score_raw']:.4f} (raw) | {case['score_percent']:.2f}%")
        elif hasattr(case, 'score') or 'score' in case:
            print(f"Score: {case.get('score', 'N/A')}")
        print()

        # Print content with better formatting
        content = case['content']
        if len(content) > 1000:
            content = content[:1000] + "..."
        print("Content:")
        print(content)
        print()

        # Print metadata
        metadata = case['metadata']
        meta_items = []
        for key, value in metadata.items():
            if key not in ['id', 'title', 'source', 'record_index'] and value:
                meta_items.append(f"{key.title()}: {value}")

        if meta_items:
            print("Metadata: " + " | ".join(meta_items))

        print("-" * 60)


def print_llm_interpretation(llm_response: str):
    """Print LLM interpretation"""
    print("\n" + "=" * 80)
    print("2) LLM CLINICAL INTERPRETATION")
    print("=" * 80)
    print()
    print(llm_response)


def print_structured_output(query: str, cases: List[Dict[str, Any]], llm_responses: List[str], vectorstore: FAISS):
    """Print all sections in a human-friendly structured layout"""
    print("\n" + "=" * 80)
    print("🔎 ZEBRA LENS — RETRIEVAL RESULTS")
    print("=" * 80)
    print(f"Query: {query}\n")

    # RAW DATABASE EXTRACTS
    print("📄 RAW DATABASE EXTRACTS\n" + "-" * 60)
    for i, case in enumerate(cases, 1):
        sim_percent = case.get("score_percent", 0.0)
        sim_raw = case.get("score_raw", 0.0)
        
        print(f"Case {i} — ID: {case['id']}  |  Similarity: {sim_percent:.2f}% ({sim_raw:.4f})")
        print(f"Title: {case['title']}")
        
        # Extract key clinical info from metadata
        metadata = case.get('metadata', {})
        age_display = metadata.get('age_display', 'Unknown')
        gender = metadata.get('gender', 'Unknown')
        pub_date = metadata.get('pub_date', 'Unknown')
        
        print(f"Demographics: {age_display}, {gender}  |  Case Date: {pub_date}")
        print()
        
        # Show content in readable paragraphs
        content = case['content'].strip()
        if not content:
            print("(No content available)")
        else:
            # Split by double newlines and show structured sections
            sections = [s.strip() for s in content.split('\n\n') if s.strip()]
            for section in sections:
                # Clean up section formatting
                if section.startswith('Title:') or section.startswith('Patient:') or section.startswith('Demographics:'):
                    print(section)
                else:
                    # Wrap long paragraphs
                    if len(section) > 800:
                        print(section[:800] + "...")
                    else:
                        print(section)
                print()
        
        # Show additional metadata if available
        meta_items = []
        for key in ['PMID', 'diagnosis', 'country']:
            value = metadata.get(key)
            if value and value != 'Unknown' and value.strip():
                meta_items.append(f"{key}: {value}")
        
        if meta_items:
            print("Additional Info: " + " | ".join(meta_items))
        
        print("-" * 60)

        # LLM INTERPRETATIONS (one per retrieved case)
        print("\n🧠 LLM CLINICAL INTERPRETATIONS (per case)\n" + "-" * 60)
        if llm_responses:
            for i, resp in enumerate(llm_responses, 1):
                print(f"Case {i} — LLM summary:")
                if resp and resp.strip():
                    paragraphs = [p.strip() for p in resp.split("\n\n") if p.strip()]
                    for para in paragraphs:
                        print(para)
                        print()
                else:
                    print("(No interpretation or error returned)")
                print("-" * 40)
        else:
            print("⚠️  No LLM interpretations available (connection failed or skipped)")
            print()

    # SYSTEM METADATA
    print("📊 SYSTEM METADATA\n" + "-" * 60)
    print(f"Retrieved cases: {len(cases)}")
    try:
        total = vectorstore.index.ntotal
        print(f"Total cases in index: {total:,}")
    except Exception:
        print("Total cases in index: Unknown")
    
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"LLM: {MODEL_NAME} @ {API_URL}")
    print(f"Search method: FAISS cosine similarity")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="ZebraLens LangChain RAG Query System")
    parser.add_argument("--query", "-q", required=True, help="Clinical query in natural language")
    parser.add_argument("--k", type=int, default=5, help="Number of cases to retrieve")
    parser.add_argument("--use-chain", action="store_true", help="Use LangChain RetrievalQA chain")
    args = parser.parse_args()

    print_header(args.query, args.k)
    
    # Test LLM connection
    connection_ok = print_connection_test()
    
    # Load vectorstore
    try:
        print(f"\nLoading vectorstore from {INDEX_DIR}...")
        vectorstore = load_vectorstore()
        print(f"✓ Loaded vectorstore with {vectorstore.index.ntotal} vectors")
    except Exception as e:
        print(f"✗ Failed to load vectorstore: {e}")
        return
    
    # Retrieve similar documents
    print(f"\nRetrieving {args.k} most similar cases...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})
    docs = retriever.get_relevant_documents(args.query)
    
    # Format cases for display
    cases = format_documents(docs)

    # Compute similarity percent scores between query and each case
    cases = compute_similarity_percent(args.query, cases)
    
    # 2) For each retrieved case, call the LLM in parallel with full user query + full case text
    llm_responses: List[str] = []
    if connection_ok and cases:
        # Prepare per-case messages (no truncation per requirement)
        messages_list = []
        for i, case in enumerate(cases, 1):
            user_msg = (
                f"User Query:\n{args.query}\n\n"
                f"Target Clinical Case (the query's patient):\nTitle: {args.query}\n\n"
                f"Retrieved Case (ID: {case['id']}):\n{case['content']}\n\n"
                "Task: Provide a concise structured summary comparing the retrieved case to the user query."
                " Cite source case ID for any findings. Do NOT provide medical advice, diagnoses, or treatment recommendations."
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            messages_list.append(messages)

        # Execute LLM calls in parallel and preserve order (response for case i -> index i)
        llm_responses = [None] * len(messages_list)
        with ThreadPoolExecutor(max_workers=min(8, len(messages_list))) as ex:
            future_to_idx = {ex.submit(call_llm_messages, m): idx for idx, m in enumerate(messages_list)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    llm_responses[idx] = fut.result()
                except Exception as e:
                    llm_responses[idx] = f"LLM Error: {e}"
    else:
        # No connection or no cases
        llm_responses = []

    # Print all results in structured, readable format
    print_structured_output(args.query, cases, llm_responses, vectorstore)


if __name__ == "__main__":
    main()
