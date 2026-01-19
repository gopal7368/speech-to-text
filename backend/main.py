from fastapi import FastAPI, UploadFile, File
import whisper
import requests
import tempfile

app = FastAPI()

# Use tiny for Windows + low RAM
model = whisper.load_model("tiny")

def call_ollama(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    return response.json().get("response", "").strip()

@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    transcript = model.transcribe(tmp_path)["text"]

    summary_prompt = f"Summarize the following meeting transcript:\n\n{transcript}"
    tasks_prompt = f"List the key action items from this meeting:\n\n{transcript}"

    summary = call_ollama(summary_prompt)
    action_items = call_ollama(tasks_prompt)

    return {
        "transcript": transcript.strip(),
        "summary": summary,
        "action_items": action_items
    }
