# LlamaIndex Tools Integration: Keenable

[Keenable](https://keenable.ai) is a web search and page-fetch API built for AI
agents. This package provides `KeenableToolSpec`, a LlamaIndex tool spec with
two tools, `search` and `fetch`, usable from any LlamaIndex agent.

**Keyless by default**: with no API key the keyless public endpoints are used.
Provide a key to use the authenticated endpoints (for higher rate limits).

## Install

```bash
pip install llama-index-tools-keenable
```

## Usage

```python
from llama_index.tools.keenable import KeenableToolSpec
from llama_index.agent.openai import OpenAIAgent

# No api_key -> keyless public endpoints. Pass api_key=... or set
# KEENABLE_API_KEY to use the authenticated endpoints.
keenable_tool = KeenableToolSpec()

agent = OpenAIAgent.from_tools(keenable_tool.to_tool_list())
agent.chat("Find recent TypeScript best practices and summarize the top result")
```

Call the tools directly:

```python
docs = keenable_tool.search("typescript best practices", site="github.com")
page = keenable_tool.fetch(docs[0].metadata["url"])
print(page[0].text)
```

`search` accepts optional per-query filters (`site`, `published_after/before`,
`acquired_after/before`, `mode`) and returns one `Document` per result. `fetch`
returns the page's main content as markdown. There is no `max_results` argument:
the API returns a fixed-size result set as-is.

## Configuration

- **API key (optional).** Constructor `api_key=...` or the `KEENABLE_API_KEY`
  environment variable. Blank → keyless public endpoints.
- **Endpoint (optional).** `KEENABLE_API_URL` overrides the base URL (HTTPS
  enforced; plain `http` only for loopback). The endpoint is never a tool
  argument the model can set, so it cannot be used to redirect requests.

The `fetch` tool rejects non-`http(s)` schemes and private/internal hosts
client-side before sending.

## License

MIT © Keenable
