#!/usr/bin/env python3
"""
ZebraLens Demo Version for Hackathon
This version allows users to either:
1. Provide their own Gemini API key (not stored permanently)
2. Use preloaded example data
"""

import os
import json
from flask import Flask, render_template, request, jsonify, session, Response

# Import from main app
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query_rag import (
    load_vectorstore, 
    format_documents, 
    compute_similarity_percent,
    call_llm_messages,
    SYSTEM_PROMPT
)
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Load preloaded demo data
PRELOADED_DATA = {}
PRELOADED_PATH = os.path.join(os.path.dirname(__file__), 'static', 'demo_preloaded_data.json')
if os.path.exists(PRELOADED_PATH):
    with open(PRELOADED_PATH, 'r', encoding='utf-8') as f:
        PRELOADED_DATA = json.load(f)
    print(f"✓ Loaded preloaded demo data: {len(PRELOADED_DATA)} queries")
else:
    print(f"⚠ Warning: Preloaded data not found at {PRELOADED_PATH}")

# Vectorstore (lazy loaded for API key mode)
vectorstore = None

def get_vectorstore():
    """Lazy load vectorstore only when needed"""
    global vectorstore
    if vectorstore is None:
        vectorstore = load_vectorstore()
    return vectorstore

def call_llm_with_user_key(messages, api_key, model="gemini-2.0-flash"):
    """Call Gemini API with user-provided API key."""
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        
        # Convert messages format
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        
        full_prompt = f"{system_content}\n\n{user_content}" if system_content else user_content
        
        response = model_instance.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=1000,
            )
        )
        
        if response.text:
            return response.text
        return "Error: Empty response from Gemini"
    except Exception as e:
        return f"Gemini Error: {str(e)}"


@app.route('/')
def index():
    """Serve demo version"""
    return render_template('index_demo.html')


@app.route('/get_demo_options')
def get_demo_options():
    """Return available preloaded demo options"""
    options = []
    for key, data in PRELOADED_DATA.items():
        options.append({
            'id': key,
            'label': data['label'],
            'query': data['query']
        })
    return jsonify({'options': options})


@app.route('/get_preloaded_results/<query_id>')
def get_preloaded_results(query_id):
    """Return preloaded results for a demo query"""
    if query_id not in PRELOADED_DATA:
        return jsonify({'error': 'Query not found'}), 404
    
    data = PRELOADED_DATA[query_id]
    return jsonify({
        'query': data['query'],
        'label': data['label'],
        'results': data['results'],
        'source': 'preloaded'
    })


@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    """Temporarily store user's API key in session"""
    data = request.json
    api_key = data.get('api_key', '')
    
    if not api_key:
        return jsonify({'error': 'No API key provided'}), 400
    
    # Validate the key by making a test call
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say 'OK' if you receive this.")
        if response.text:
            session['gemini_api_key'] = api_key
            return jsonify({'success': True, 'message': 'API key validated and set'})
        return jsonify({'error': 'Invalid API key - no response'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid API key: {str(e)}'}), 400


@app.route('/search_live', methods=['POST'])
def search_live():
    """Live search using user's API key"""
    data = request.json
    query = data.get('query', '')
    api_key = session.get('gemini_api_key')
    
    if not api_key:
        return jsonify({'error': 'No API key set. Please provide your Gemini API key.'}), 401
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        vs = get_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(query)
        
        cases = format_documents(docs)
        cases = compute_similarity_percent(query, cases)
        cases.sort(key=lambda x: x.get('score_percent', 0), reverse=True)
        
        # Process top 5 cases with user's API key
        results = []
        for case in cases[:5]:
            user_msg = (
                f"User Query:\n{query}\n\n"
                f"Retrieved Case (ID: {case['id']}):\n{case['content']}\n\n"
                "Task: Provide a concise structured summary comparing the retrieved case to the user query."
                " Cite source case ID for any findings. Do NOT provide medical advice."
            )
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            
            llm_response = call_llm_with_user_key(messages, api_key)
            
            metadata = case.get('metadata', {})
            result = {
                'id': case['id'],
                'title': case['title'],
                'age': metadata.get('age_display', 'Unknown'),
                'gender': metadata.get('gender', 'Unknown'),
                'pub_date': metadata.get('pub_date', 'Unknown'),
                'similarity_percent': case.get('score_percent', 0),
                'llm_response': llm_response,
                'pmid': metadata.get('PMID', ''),
                'content': case['content'][:2000]
            }
            results.append(result)
        
        return jsonify({
            'query': query,
            'results': results,
            'source': 'live'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat_with_case_demo', methods=['POST'])
def chat_with_case_demo():
    """Chat with a case using user's API key"""
    data = request.json
    case_id = data.get('case_id')
    question = data.get('question', '')
    case_content = data.get('case_content', '')
    api_key = session.get('gemini_api_key')
    
    if not api_key:
        return jsonify({'error': 'No API key set'}), 401
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Case {case_id}:\n{case_content}\n\nQuestion: {question}"},
    ]
    
    response = call_llm_with_user_key(messages, api_key)
    return jsonify({'answer': response})


# ========== GEOCODING SUPPORT ==========
from geopy.geocoders import Nominatim
from functools import lru_cache
import re

geolocator = Nominatim(user_agent="zebralens_demo_v1")

@lru_cache(maxsize=500)
def geocode_location(location_name):
    """Geocode a location name to coordinates."""
    if not location_name:
        return None
    try:
        location = geolocator.geocode(location_name, timeout=5)
        if location:
            return {'lat': location.latitude, 'lon': location.longitude}
        # Fallback
        parts = location_name.split(',')
        if len(parts) > 1:
            shorter = ", ".join(parts[-2:]).strip()
            location = geolocator.geocode(shorter, timeout=5)
            if location:
                return {'lat': location.latitude, 'lon': location.longitude}
        return None
    except Exception:
        return None


def extract_location_heuristic(text):
    """Extract location from case text."""
    if not text:
        return None
    
    countries = [
        "USA", "United States", "UK", "United Kingdom", "China", "India", "Japan", "Brazil", 
        "Germany", "France", "Italy", "Spain", "Canada", "Australia", "Mexico", "Thailand",
        "Vietnam", "Korea", "Taiwan", "Turkey", "Iran", "Egypt", "South Africa", "Nigeria",
        "Kenya", "Argentina", "Colombia", "Russia", "Netherlands", "Sweden", "Switzerland"
    ]
    
    cities = {
        "Barcelona": "Spain", "Madrid": "Spain", "Paris": "France", "London": "UK",
        "Berlin": "Germany", "Rome": "Italy", "New York": "USA", "Boston": "USA",
        "Tokyo": "Japan", "Seoul": "Korea", "Beijing": "China", "Shanghai": "China"
    }
    
    text_sample = text[:2000]
    for term in countries + list(cities.keys()):
        if re.search(r'\b' + re.escape(term) + r'\b', text_sample, re.IGNORECASE):
            return cities.get(term, term)
    return None


@app.route('/geocode_country', methods=['GET'])
def geocode_country():
    """Geocode a country name to coordinates"""
    country = request.args.get('country', '')
    if not country:
        return jsonify({'error': 'No country provided'}), 400
    
    coords = geocode_location(country)
    if coords:
        return jsonify(coords)
    return jsonify({'error': f'Could not geocode: {country}'}), 404


@app.route('/query_stream', methods=['POST'])
def query_stream():
    """Live search using user's API key with SSE streaming format"""
    data = request.json
    query = data.get('query', '')
    api_key = session.get('gemini_api_key')
    
    if not api_key:
        return jsonify({'error': 'No API key set. Please provide your Gemini API key.'}), 401
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    def generate_stream():
        try:
            print("STREAM: Generator started")
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Connecting to Gemini AI...'})}\n\n"
            
            # Load vectorstore
            vs = get_vectorstore()
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching medical case database...'})}\n\n"
            
            # Retrieve cases
            retriever = vs.as_retriever(search_kwargs={"k": 50})
            docs = retriever.invoke(query)
            
            cases = format_documents(docs)
            cases = compute_similarity_percent(query, cases)
            cases.sort(key=lambda x: x.get('score_percent', 0), reverse=True)
            
            # Send metadata
            yield f"data: {json.dumps({'type': 'metadata', 'query': query, 'k': 10, 'found_cases': len(cases[:10])})}\n\n"
            
            # Prepare cases for map (with locations)
            map_cases = []
            for case in cases[:20]:
                location = extract_location_heuristic(case['content'])
                coords = geocode_location(location) if location else None
                map_cases.append({
                    'id': case['id'],
                    'title': case['title'],
                    'similarity_percent': case.get('score_percent', 0),
                    'location': location,
                    'location_coords': coords
                })
            
            # Send initial locations for map
            yield f"data: {json.dumps({'type': 'initial_locations', 'cases': map_cases})}\n\n"
            
            # Send panel cases (top 10 for placeholders)
            panel_cases = [{'id': c['id'], 'title': c['title']} for c in cases[:10]]
            yield f"data: {json.dumps({'type': 'panel_cases', 'cases': panel_cases})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating AI analysis for each case...'})}\n\n"
            
            # Process each case with LLM
            for idx, case in enumerate(cases[:10]):
                user_msg = (
                    f"User Query:\n{query}\n\n"
                    f"Retrieved Case (ID: {case['id']}):\n{case['content']}\n\n"
                    "Task: Provide a concise structured summary comparing the retrieved case to the user query."
                )
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                
                llm_response = call_llm_with_user_key(messages, api_key)
                
                metadata = case.get('metadata', {})
                location = extract_location_heuristic(case['content'])
                location_coords = geocode_location(location) if location else None
                
                result = {
                    'id': case['id'],
                    'title': case['title'],
                    'age': metadata.get('age_display', 'Unknown'),
                    'gender': metadata.get('gender', 'Unknown'),
                    'pub_date': metadata.get('pub_date', 'Unknown'),
                    'similarity_percent': case.get('score_percent', 0),
                    'llm_response': llm_response,
                    'pmid': metadata.get('PMID', ''),
                    'content': case['content'][:2000],
                    'location': location,
                    'location_coords': location_coords,
                    'location_heuristic': location
                }
                
                # Stream each case result
                yield f"data: {json.dumps({'type': 'case', 'case_index': idx, 'result': result})}\n\n"
            
            # Send completion
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    print("STREAM: Starting streaming response")
    response = Response(generate_stream(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response


@app.route('/healthz')
def healthz():
    """Health check endpoint for Render"""
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print("\n" + "="*60)
    print("ZebraLens DEMO VERSION")
    print("="*60)
    print(f"Preloaded queries available: {len(PRELOADED_DATA)}")
    print(f"\nStarting server on http://localhost:{port}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)

