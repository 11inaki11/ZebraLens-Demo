# 🦓 ZebraLens Demo

**AI-Powered Rare Disease Case Matching**

ZebraLens uses semantic search and AI analysis to help medical professionals find similar rare disease cases from a database of published case reports.

## Features

- 🔍 **Semantic Search**: Find similar cases based on clinical descriptions
- 🗺️ **Global Map Visualization**: See cases plotted on an interactive 3D globe
- 🤖 **AI Analysis**: Get detailed comparisons powered by Gemini AI
- 📊 **Pre-loaded Examples**: Explore demo queries without an API key

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the demo server:
```bash
python app_demo.py
```

3. Open http://localhost:5002 in your browser

### Using the Demo

**Option 1: Pre-loaded Examples**
- Select "Use Preloaded Examples" to explore pre-generated case searches
- No API key required

**Option 2: Live Search**
- Select "Use Your Own Gemini API Key"
- Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Enter your key and perform live searches

## Technology Stack

- **Backend**: Flask (Python)
- **AI**: Google Gemini 2.0 Flash
- **Vector Search**: FAISS with SentenceTransformers
- **Map**: CesiumJS for 3D globe visualization
- **Frontend**: Vanilla JavaScript with modern CSS

## License

MIT License - See LICENSE file for details.

## Hackathon Project

Built for the Google AI Hackathon 2026.
