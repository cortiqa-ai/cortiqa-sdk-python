"""Streaming: Stream tokens in real time from Cortiqa Falin."""

import os
import sys
from cortiqa import Cortiqa

api_key = os.environ.get("CORTIQA_API_KEY", "sk-cortiqa-your-api-key")
client = Cortiqa(api_key=api_key)

def main():
    print("Streaming response from falin-01:\n")

    stream = client.chat.completions.create(
        model="falin-01",
        messages=[
            {"role": "user", "content": "Write a 4-line poem about the Indian monsoon."}
        ],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            sys.stdout.write(chunk.choices[0].delta.content)
            sys.stdout.flush()

    print("\n\nDone streaming!")

if __name__ == "__main__":
    main()
