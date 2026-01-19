# Speech-to-Text AI Meeting Notes Generator (Whisper + LLaMA2)

This application allows users to upload a recorded meeting and automatically
generates:
- A concise summary
- Key action items
- A full transcript

## Tech Stack
- Whisper (local transcription)
- LLaMA2 via Ollama (summarization & task extraction)
- FastAPI backend
- Streamlit frontend

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Pull the model:
   ollama pull llama2
3. Start backend:
   uvicorn backend.main:app --reload
4. Start frontend:
   streamlit run frontend/app.py
