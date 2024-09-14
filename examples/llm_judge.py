import asyncio
import csv
import json

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import print

from simple_ai_agents.chat_agent import ChatAgentAsync
from simple_ai_agents.models import LLMOptions

load_dotenv()

gpt4o: LLMOptions = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
}

mistral_small: LLMOptions = {
    "model": "mistral/mistral-small",
    "temperature": 0.7,
}

mistral_medium: LLMOptions = {
    "model": "mistral/mistral-medium",
    "temperature": 0.7,
}

gpt4: LLMOptions = {
    "model": "gpt-4-1106-preview",
    "temperature": 0.7,
}

# Single grading prompt "single-v1-multi-turn" from LLM-as-a-Judge paper
grading_prompt = """Please act as an impartial judge and evaluate the quality of the response
provided by an AI assistant to the user question displayed below. Your evaluation should consider factors
such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
the response. Begin your evaluation by providing a short explanation. Be as objective as
possible. After providing your explanation, please rate the response on a scale of 1 to 10."""  # noqa: E501

evaluation_prompt = """[Question Part 1]
{q1}
[The Start of Assistant’s Answer Part 1]
{r1}
[The End of Assistant’s Answer Part 1]
[Question Part 2]
{q2}
[The Start of Assistant’s Answer Part 2]
{r2}
[The End of Assistant’s Answer Part 2]
"""


def read_jsonl(category_filter):
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"  # noqa: E501
    response = requests.get(url)  # jsonl file
    response.raise_for_status()

    result = []
    for line in response.text.splitlines():
        data = json.loads(line)
        if data.get("category") == category_filter:
            result.append(data)

    return result


class Grading(BaseModel):
    """Grading of a response"""

    explanation: str = Field(description="Evaluation of response quality")
    grade: int = Field(description="Grade given to the response", ge=1, le=10)


async def call_agent(llm_options, question):
    q1, q2 = question[0], question[1]
    print(f"{llm_options['model']} Started")
    agent = ChatAgentAsync(llm_options=llm_options)
    r1 = await agent(q1)
    r2 = await agent(q2)
    judge = ChatAgentAsync(llm_options=gpt4)
    result = await judge.gen_model(
        evaluation_prompt.format(q1=q1, q2=q2, r1=r1, r2=r2),
        response_model=Grading,
    )
    print(f"{llm_options['model']} Completed")
    return result, r1, r2, agent


async def comparison(questions):
    """
    Compare the responses from two different models with grading by a GPT-4 judge.
    """
    output = []
    models = [gpt4o, mistral_small, mistral_medium]
    for data in questions:
        question = data["turns"]
        print(question)
        tasks = [call_agent(opt, question) for opt in models]
        results = await asyncio.gather(*tasks)
        output.append(results)
    # Write output to csv
    with open("model_grading_example.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Question",
                "GPT-3",
                "GPT-3 Eval",
                "GPT-3 Grade",
                "Mistral-Small",
                "Mistral-Small Eval",
                "Mistral-Small Grade",
                "Mistral-Medium",
                "Mistral-Medium Eval",
                "Mistral-Medium Grade",
            ]
        )
        for i, data in enumerate(output):
            writer.writerow(
                [
                    "\n".join(questions[i]["turns"]),
                    data[0][1] + "\n" + data[0][2],
                    data[0][0].explanation,
                    data[0][0].grade,
                    data[1][1] + "\n" + data[1][2],
                    data[1][0].explanation,
                    data[1][0].grade,
                    data[2][1] + "\n" + data[2][2],
                    data[2][0].explanation,
                    data[2][0].grade,
                ]
            )


if __name__ == "__main__":
    questions = read_jsonl("writing")
    asyncio.run(comparison(questions))
