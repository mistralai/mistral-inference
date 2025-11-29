"""
Example: Safe streaming JSON parsing using OpenRouter with Mistral API

This script demonstrates how to safely handle streamed responses from OpenRouter
by parsing JSON chunks line-by-line and ignoring malformed lines.
"""

import json
import requests

API_KEY = "sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/ktdjiren",  
    "X-Title": "Mistral JSON Stream Parser"
}

def safe_parse_json(line: str):
    """Attempts to parse a JSON line safely. Logs any errors."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None

def call_mistral_api():
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": True
    }

    try:
        response = requests.post(ENDPOINT, headers=HEADERS, json=payload, stream=True)

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue  # Skip empty or malformed lines

            json_chunk = line.removeprefix("data:").strip()
            parsed = safe_parse_json(json_chunk)

            if parsed:
                delta = parsed.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    print(delta["content"], end="", flush=True)

    except Exception as e:
        print("Request failed:", e)

if __name__ == "__main__":
    call_mistral_api()
