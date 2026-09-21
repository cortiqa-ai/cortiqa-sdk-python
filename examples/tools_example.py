"""Tool / Function calling example with Cortiqa Falin."""

import json
from cortiqa import Cortiqa

client = Cortiqa()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_stock_price",
            "description": "Get current stock price and change for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. TATAMOTORS, RELIANCE)"},
                },
                "required": ["ticker"],
            },
        },
    }
]

def lookup_stock_price(ticker: str) -> str:
    # Simulated execution
    return json.dumps({"ticker": ticker, "price": "₹980.50", "change": "+2.4%"})

def main():
    messages = [
        {"role": "user", "content": "What is the current stock price of TATAMOTORS?"}
    ]

    print("Sending prompt with tools to falin-01...")
    response = client.messages.create(
        model="falin-01",
        messages=messages,
        tools=tools,
    )

    choice = response.choices[0]
    if choice.message.tool_calls:
        for tool_call in choice.message.tool_calls:
            print(f"\nModel requested Tool Call: {tool_call.function.name}")
            args = json.loads(tool_call.function.arguments)
            print(f"Arguments: {args}")

            # Execute tool
            tool_result = lookup_stock_price(**args)

            # Append assistant message and tool response
            messages.append(choice.message.model_dump())
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        print("\nSynthesizing final answer with tool result...")
        final_answer = client.messages.create(
            model="falin-01",
            messages=messages,
        )
        print("\nFinal Output:")
        print(final_answer.content)
    else:
        print(response.content)

if __name__ == "__main__":
    main()
