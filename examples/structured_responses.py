from dotenv import load_dotenv
from pydantic import BaseModel, Field

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.models import LLMOptions

load_dotenv()

openai: LLMOptions = {"model": "gpt-4o-mini", "temperature": 0.7}
together: LLMOptions = {
    "model": "together_ai/togethercomputer/Meta-Llama-3.1-70B-Instruct-Turbo",
    "temperature": 0.7,
}
llama3: LLMOptions = {
    "model": "ollama/llama3.1",
    "temperature": 0.7,
    "api_base": "http://localhost:11434",
}


class Ingredient(BaseModel):
    """Ingredient used in the recipe"""

    name: str = Field(description="Name of the ingredient")
    weight: int = Field(description="Weight of the ingredient used in grams")


class Recipe(BaseModel):
    """A recipe for a dish"""

    recipe_name: str = Field(description="Name of the recipe")
    serving: int = Field(description="Number of servings")
    ingredients: list[Ingredient]


def gen_recipe(recipe_name: str):
    """
    AI generated recipe. We use openai to generate the recipe in a structured format
    and a local mistral model to generate the tips.

    Note: structured response is not added to the history and should be added to
    subsequent prompts as required.
    """
    chatbot = ChatAgent(
        system="You are an Italian chef", llm_options=openai, character="Chef"
    )
    recipe = chatbot.gen_model(
        f"Generate a {recipe_name} recipe", response_model=Recipe, llm_options=together
    )
    chatbot(
        f"Provide some helpful tips to cook this dish:\n {recipe}",
        console_output=True,
    )
    return recipe


if __name__ == "__main__":
    # Since pasta is a pydantic model, we can easily add it to a database or serve it as an API
    pasta = gen_recipe("aglio olio")
