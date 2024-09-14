from dotenv import load_dotenv
from rich import print

from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import LLMOptions

load_dotenv()

# Depending on the selected provider, you might need to set the API key
# See https://docs.litellm.ai/docs/providers for list of providers
# See https://docs.litellm.ai/docs/completion/input#input-params-1 for list of options
# Note: message will be automatically prepared using the input prompt
openai: LLMOptions = {"model": "gpt-4o-mini", "temperature": 0.7}
llama: LLMOptions = {
    "model": "huggingface/meta-llama/Llama-2-13b-chat-hf",
    "temperature": 0.5,
}
mistral: LLMOptions = {
    "model": "ollama/mistral",
    "temperature": 0.7,
    "api_base": "http://localhost:11434",
}

default_options = openai


def chat():
    sess = ChatLLMSession()
    # gen, gen_async, stream and stream_async
    response = sess.gen("Write a haiku about trees", llm_options=default_options)
    print(response)
    score = sess.gen("Rate the haiku from 1-10", llm_options=default_options)
    print(score)


if __name__ == "__main__":
    chat()

    # Silent sentinels,
    # Branches sway in gentle breeze,
    # Nature's gift of peace.
    # As an AI, I don't possess personal opinions or emotions.
    # However, I can say that your haiku is well-written and
    # captures the essence of trees in a concise manner.
    # It would be fair to give it a rating of 8 or 9 out of 10.
