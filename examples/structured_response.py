from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import print

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.models import LLMOptions

load_dotenv()

openai: LLMOptions = {"model": "gpt-3.5-turbo", "temperature": 0.7}
mistral: LLMOptions = {
    "model": "ollama/mistral",
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


def recipe():
    """
    AI generated recipe. We use openai to generate the recipe in a structured format
    and a local mistral model to generate the tips.

    Note: structured response is not added to the history and should be added to
    subsequent prompts as required.
    """
    chatbot = ChatAgent(
        system="You are an Italian chef", llm_options=mistral, character="Chef"
    )
    recipe = chatbot.gen_model(
        "Generate a aglio olio recipe", response_model=Recipe, llm_options=openai
    )
    print(recipe)
    chatbot(
        f"Provide some helpful tips to cook this dish:\n {recipe}",
        console_output=True,
    )


if __name__ == "__main__":
    recipe()
