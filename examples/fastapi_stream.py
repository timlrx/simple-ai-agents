from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import LLMOptions

# To run the FastAPI server, execute the following command:
# fastapi run examples/fastapi_stream.py

# Once the server is up, try running the following script:
# ```python
# import httpx
# url = "http://localhost:8000"
# with httpx.stream("POST", url, json={"prompt": "why do stars twinkle?"}) as r:
#     for chunk in r.iter_raw():  # or, for line in r.iter_lines():
#         print(chunk)
# ```

load_dotenv()
openai: LLMOptions = {"model": "gpt-4o-mini", "temperature": 0.7}
github: LLMOptions = {
    "model": "github/gpt-4o-mini",
}
default_options = github

app = FastAPI()


async def generate_response(prompt: str) -> AsyncGenerator[str, None]:
    """Generate the AI model response and stream it."""
    sess = ChatLLMSession()
    stream = sess.stream_async(prompt, llm_options=default_options)
    async for chunk in stream:
        yield chunk["delta"]


@app.post("/")
async def query(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "Write a haiku about trees")
    return StreamingResponse(generate_response(prompt), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
