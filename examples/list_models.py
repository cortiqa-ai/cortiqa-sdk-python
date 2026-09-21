"""Inspect all available models from Cortiqa."""

import os
from cortiqa import Cortiqa

def main():
    api_key = os.environ.get("CORTIQA_API_KEY", "sk-cortiqa-your-api-key")
    client = Cortiqa(api_key=api_key)

    print("Fetching available Cortiqa AI Models...")
    models = client.models.list()

    print(f"\nFound {len(models)} models:")
    print("-" * 60)
    for model in models:
        tier = "Free Tier" if model.free else "Pro/Enterprise"
        print(f"ID:          {model.id} ({model.name or model.id})")
        print(f"Provider:    {model.provider}")
        print(f"Access:      {tier}")
        print(f"Description: {model.description}")
        print("-" * 60)

if __name__ == "__main__":
    main()
