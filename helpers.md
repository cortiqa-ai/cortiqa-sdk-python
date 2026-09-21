# Cortiqa SDK Helpers & Advanced Usage

This guide covers advanced developer utilities, streaming helpers, network resilience, proxy configuration, and debugging.

---

## 1. Streaming Helpers

### The `stream()` Context Manager

The `client.messages.stream()` context manager automatically collects generated tokens, exposes an effortless `.text_stream` iterator, and builds the full `ChatCompletion` object upon completion:

```python
from cortiqa import Cortiqa

client = Cortiqa()

with client.messages.stream(
    model="falin-01",
    messages=[{"role": "user", "content": "Write a 5-step morning routine."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

    # Access full message after streaming ends
    final_message = stream.get_final_message()
    print(f"\nFinished model: {final_message.model}")
```

### Async Streaming

```python
import asyncio
from cortiqa import AsyncCortiqa

async def main():
    async with AsyncCortiqa() as client:
        stream = client.messages.stream(
            model="falin-01",
            messages=[{"role": "user", "content": "Tell me a joke."}]
        )
        async with stream as s:
            async for token in s.text_stream():
                print(token, end="", flush=True)

asyncio.run(main())
```

---

## 2. Automatic Retries & Network Resilience

The SDK automatically retries requests that fail due to:
- **Rate Limits (`429 Too Many Requests`)**
- **Transient Server Errors (`500`, `502`, `503`, `504`)**
- **Network connection resets or disconnects**

It uses exponential backoff with jitter to prevent server stampedes.

You can configure the retry count during client initialization:

```python
# Retry up to 4 times
client = Cortiqa(max_retries=4)

# Disable automatic retries completely
client = Cortiqa(max_retries=0)
```

---

## 3. Timeouts

By default, requests time out after 60 seconds. You can customize the timeout globally or per client:

```python
# 15 seconds global timeout
client = Cortiqa(timeout=15.0)
```

---

## 4. Custom HTTP Client (Proxies, SSL, Connection Pools)

You can pass your own configured `httpx.Client` or `httpx.AsyncClient` to use custom corporate proxies, mutual TLS, or connection pool limits:

```python
import httpx
from cortiqa import Cortiqa

# Configure proxy
custom_http = httpx.Client(
    proxy="http://corp-proxy.internal:8080",
    verify="/path/to/internal-ca.pem",
    limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
)

client = Cortiqa(http_client=custom_http)
```

---

## 5. Debug Logging

To log all outgoing HTTP requests, headers, and responses:

```python
import logging

# Enable HTTPX debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

from cortiqa import Cortiqa

client = Cortiqa()
client.messages.create(
    model="falin-01",
    messages=[{"role": "user", "content": "ping"}]
)
```
