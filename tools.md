# Tool Calling with Cortiqa Falin Models

Cortiqa models (such as `falin-01` and `falin-pro`) support structured **Tool Calling** (Function Calling), allowing the model to invoke custom functions, call external APIs, query databases, or perform agentic workflows.

---

## How Tool Calling Works

```
1. Client sends messages + tool definitions
2. Model decides to call a tool and returns `tool_calls`
3. Client executes the function locally
4. Client sends tool result back with `role: "tool"`
5. Model generates the final natural language answer
```

---

## Defining Tools

Tools are provided as a list of dictionaries adhering to the JSON Schema standard:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city and state, e.g. Mumbai, IN",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }
]
```

---

## Complete Example

```python
import json
from cortiqa import Cortiqa

client = Cortiqa()

# 1. Define your real Python function
def get_weather(city: str, unit: str = "celsius") -> str:
    # Simulated weather lookup
    return json.dumps({"city": city, "temperature": "29°C", "condition": "Sunny"})

# 2. Define tools schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

# 3. Call Cortiqa with tools
messages = [
    {"role": "user", "content": "What is the weather like in Mumbai right now?"}
]

response = client.messages.create(
    model="falin-01",
    messages=messages,
    tools=tools,
)

choice = response.choices[0]

# 4. Check if the model decided to call a tool
if choice.message.tool_calls:
    for tool_call in choice.message.tool_calls:
        func_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print(f"Model requested calling: {func_name} with {arguments}")

        # Execute local function
        if func_name == "get_weather":
            tool_output = get_weather(**arguments)

            # Append assistant message with tool call
            messages.append(choice.message.model_dump())

            # Append tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output
            })

    # 5. Send results back to model for final synthesis
    final_response = client.messages.create(
        model="falin-01",
        messages=messages,
    )

    print("\nFinal Answer:")
    print(final_response.content)
```

---

## Tool Choice Options

You can control how the model decides to use tools using `tool_choice`:

* `"auto"`: Default behavior. The model decides whether to respond with text or call one or more tools.
* `"required"`: Forces the model to call at least one tool.
* Specific tool: Forces the model to call a particular tool:
  ```python
  tool_choice = {"type": "function", "function": {"name": "get_weather"}}
  ```
