import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

load_dotenv()


os.environ["AZURE_OPENAI_API_KEY"] =  os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
model = "gpt-4o-mini"
version = "2024-12-01-preview"

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment=model, 
    api_version=version,
    temperature=0.78,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

if __name__ == "__main__":

    # standard invokation

    from langchain_core.messages.utils import get_buffer_string

    print("running standard invokation...\n\n")
    messages = [
        SystemMessage(
            content="You are a helpful assistant! Your name is Bob."
        ),
        HumanMessage(
            content="What is your name?"
        )
    ]
    messages.append(AIMessage(content=llm.invoke(messages).content))
    print(get_buffer_string(messages), "\n\n")

    # structured invokation

    from typing_extensions import TypedDict, List
    from pydantic import BaseModel, Field

    print("running structured invokation...\n\n")

    class ModelSchema(BaseModel):
        names: str = Field(
            description="A commar seperated list of names.",
        )
        number: List[int] = Field(
            description="A list of the number of letters in each name"
        )
    
    messages = [
        SystemMessage(
            content="You are a helpful assistant! Your name is Bob."
        ),
        HumanMessage(
            content="Give me some boys name for Italian kids."
        )
    ]

    response = llm.with_structured_output(ModelSchema).invoke(messages)
    print(f'List of names: {response.names}.')
    print(f'number of names: {response.number}.')



