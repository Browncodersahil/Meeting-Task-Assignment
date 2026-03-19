# Meeting-Task-Assignment

An intelligent system that automatically extracts tasks from meeting audio recordings and assigns them to the right team members using NLP and skill-based matching.

## Drive Link Of Task-Assignment Video 
Link-- https://docs.google.com/videos/d/1WETh_MI_ffLEruJ6k4X3_roHk75K1lKBCiUHI6DhWAA/edit?usp=sharing

## How It Works

```
Audio File → Groq Whisper API → Transcript → spaCy NLP Engine → tasks.json
```

1. **Speech to Text** — Audio is transcribed using Groq's Whisper API
2. **Task Extraction** — Custom NLP engine identifies actionable tasks from the transcript
3. **Smart Assignment** — Tasks are assigned based on who is mentioned, their role, and their skills
4. **Output** — Results are saved as a structured JSON file and displayed as an ASCII table

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/meeting-task-assignment.git
cd meeting-task-assignment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure API key
Create a `.env` file in the project root:
```
GROQ_API_KEY='your_groq_api_key_here'
```
Get a free API key at https://console.groq.com

### 4. Configure your team
Edit `team.json` with your team members:
```json
[
  {
    "name": "John",
    "role": "ML Engineer",
    "skills": ["machine learning", "model", "training", "data", "pipeline"]
  }
]
```

## Usage

```bash
python main.py --audio meeting.mp3 --team team.json
```

Output is saved to `output/<audio_filename>.json`

### Supported Audio Formats
`.wav` `.mp3` `.m4a` `.ogg` `.flac` `.mp4`

## Tech Stack

| Component | Technology |
|---|---|
| Speech to Text | Groq Whisper API (whisper-large-v3) |
| NLP | spaCy (en_core_web_sm) |
| Language | Python 3.11+ |

## Design Decisions

- **No regex** — all text processing uses spaCy's tokenizer and dependency parser
- **No AI for task logic** — spaCy is only used for grammar parsing; all task identification and assignment logic is custom rule-based code
- **Groq over local Whisper** — no ffmpeg dependency, faster transcription, free tier available
- **Skill scoring over simple name matching** — handles STT name errors and unassigned tasks gracefully
