# Changelog

All notable changes to the `cortiqa` SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-09-21

### Added
- Initial release of the official Cortiqa Python SDK.
- **Synchronous and Asynchronous Clients**: `Cortiqa` and `AsyncCortiqa` powered by HTTPX.
- **Dual API Syntax**:
  - Anthropic-style: `client.messages.create(...)` and `client.messages.stream(...)`.
  - OpenAI-style: `client.chat.completions.create(...)`.
- **Falin Models Support**: Seamless inference with `falin-01`, `falin-pro`, `falin-vision`, and `falin-ultra`.
- **Tool Calling (Function Calling)**: Full support for passing tools, functions, and receiving structured tool execution calls.
- **Streaming Helpers**: `StreamManager` with `.text_stream` iterator and `.get_final_message()` helper.
- **Resilience**: Built-in automatic exponential backoff retries on rate limits (429) and server errors (5xx).
- **Strong Typing**: Pydantic v2 data models for request and response payloads.
- Comprehensive test suite and quickstart examples.
