import argparse
import json
import os
import sys

from NLP_Engine import TeamMember, TaskExtractor, TaskAssigner, OutputFormatter

SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4"}


def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip().strip("'").strip('"')

_load_env()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


class WhisperTranscriber:

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key missing. Add GROQ_API_KEY to your .env file.")
        from groq import Groq
        self._client = Groq(api_key=api_key)

    def transcribe(self, audio_path: str) -> str:
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}")

        print(f"[Groq Whisper] Transcribing: {audio_path}")
        with open(audio_path, "rb") as f:
            transcript = self._client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        print(f"[Groq Whisper] Done. {len(transcript)} characters transcribed.")
        return transcript.strip()


def load_team(path: str) -> list[TeamMember]:
    with open(path, "r") as f:
        data = json.load(f)
    return [TeamMember(m["name"], m["role"], m.get("skills", [])) for m in data]


def run(audio_path: str, team_path: str, output_path: str):
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    if not os.path.exists(team_path):
        print(f"ERROR: Team file not found: {team_path}")
        sys.exit(1)

    team = load_team(team_path)

    transcriber = WhisperTranscriber(api_key=GROQ_API_KEY)
    transcript  = transcriber.transcribe(audio_path)

    txt_path = os.path.splitext(audio_path)[0] + "_transcript.txt"
    with open(txt_path, "w") as f:
        f.write(transcript)
    print(f"[Groq Whisper] Transcript saved → {txt_path}")

    print("\n" + "=" * 65)
    print("  MEETING TASK ASSIGNMENT SYSTEM")
    print("=" * 65)

    print("\n[Step 1/3] Extracting tasks...")
    extractor = TaskExtractor(team)
    tasks     = extractor.process(transcript)
    print(f"           {len(tasks)} tasks found.")

    print("[Step 2/3] Assigning tasks...")
    assigner = TaskAssigner(team)
    tasks    = assigner.process(tasks)

    print("[Step 3/3] Formatting output...")
    formatter = OutputFormatter()
    formatter.process(tasks, output_path=output_path)

    print(f"\n✅ Done! JSON saved → {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Meeting Task Assignment System")
    parser.add_argument("--audio",  required=True, help="Path to audio file")
    parser.add_argument("--team",   required=True, help="Path to team JSON file")
    parser.add_argument("--output", default=None, help="Output JSON path (default: output/<audio_name>.json)")
    args = parser.parse_args()

    audio_name  = os.path.splitext(os.path.basename(args.audio))[0]
    output_path = args.output or f"output/{audio_name}.json"

    run(args.audio, args.team, output_path)


if __name__ == "__main__":
    main()