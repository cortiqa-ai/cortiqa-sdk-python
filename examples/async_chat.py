"""Asynchronous chat completion using AsyncCortiqa."""

import os
import asyncio
from cortiqa import AsyncCortiqa

async def main():
    api_key = os.environ.get("CORTIQA_API_KEY", "sk-cortiqa-your-api-key")

    async with AsyncCortiqa(api_key=api_key) as client:
        print("Sending async request to falin-pro...")
        response = await client.chat.completions.create(
            model="falin-pro",
            messages=[
                {"role": "user", "content": "Write an efficient Python fibonacci generator using yield."}
            ]
        )

        print("\nGenerated Code:")
        print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
