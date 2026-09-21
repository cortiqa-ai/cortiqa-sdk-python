"""Stream context manager example (Anthropic style)."""

from cortiqa import Cortiqa

client = Cortiqa()

def main():
    print("Streaming tokens with context manager:\n")

    with client.messages.stream(
        model="falin-01",
        messages=[{"role": "user", "content": "List 3 healthy habits for software developers."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

        final_msg = stream.get_final_message()
        print(f"\n\nStream complete! Accumulated {len(stream._collected_tokens)} chunks.")

if __name__ == "__main__":
    main()
