"""Quickstart: Send a chat completion request to Cortiqa Falin-01."""

import os
from cortiqa import Cortiqa

# Initialize client (uses CORTIQA_API_KEY environment variable or direct parameter)
api_key = os.environ.get("CORTIQA_API_KEY", "sk-cortiqa-your-api-key")
client = Cortiqa(api_key=api_key)

def main():
    print("Sending prompt to Cortiqa Falin-01...")
    response = client.chat.completions.create(
        model="falin-01",
        messages=[
            {"role": "system", "content": "You are a helpful and concise AI assistant built by Cortiqa."},
            {"role": "user", "content": "Explain artificial general intelligence in 2 sentences."}
        ],
        temperature=0.7,
        max_tokens=250,
    )

    print("\nResponse:")
    print(response.choices[0].message.content)

    if response.usage:
        print(f"\nTokens used: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")

if __name__ == "__main__":
    main()
