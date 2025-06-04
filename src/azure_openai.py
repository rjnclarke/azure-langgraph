import os
from typing import List, Dict, Union
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.messages.utils import get_buffer_string

# Load .env variables
load_dotenv()
api_version = "2024-12-01-preview"
deployment = "gpt-4o-mini"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)

def message_to_dict(lstr: List[BaseMessage]) -> List[Dict[str, str]]:
    """
    Convert a list of LangChain BaseMessage objects to a list of OpenAI-compatible message dicts.

    Args:
        lstr (List[BaseMessage]): List of messages, each a HumanMessage, AIMessage, or SystemMessage.

    Returns:
        List[Dict[str, str]]: Each dict contains keys 'role' and 'content', compatible with OpenAI Chat Completions API.

    Raises:
        ValueError: If an unexpected message type is encountered in the list.
    """
    out = []
    for el in lstr:
        if isinstance(el, HumanMessage):
            out.append({"role": "user", "content": el.content})
        elif isinstance(el, AIMessage):
            out.append({"role": "assistant", "content": el.content})
        elif isinstance(el, SystemMessage):
            out.append({"role": "system", "content": el.content})
        else:
            raise ValueError(f"Unknown message type: {type(el)}")
    return out

def dict_to_message(lstr: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    Convert a list of message dictionaries (role/content) to LangChain BaseMessage objects.

    Args:
        lstr (List[Dict[str, str]]): List of dictionaries, each with 'role' and 'content' keys.

    Returns:
        List[BaseMessage]: Corresponding list of HumanMessage, AIMessage, or SystemMessage objects.

    Raises:
        ValueError: If a dict contains an unknown 'role'.
    """
    out = []
    for el in lstr:
        if el["role"] == "user":
            out.append(HumanMessage(content=el["content"]))
        elif el["role"] == "assistant":
            out.append(AIMessage(content=el["content"]))
        elif el["role"] == "system":
            out.append(SystemMessage(content=el["content"]))
        else:
            raise ValueError(f"Unknown message role: {el['role']}")
    return out

def get_response(messages: Union[List[BaseMessage], List[Dict[str, str]]]) -> List[BaseMessage]:
    """
    Get a chatbot response using Azure OpenAI, appending the reply to the conversation.

    Args:
        messages (Union[List[BaseMessage], List[Dict[str, str]]]):
            Conversation history as BaseMessage objects or as dicts with keys 'role' and 'content'.

    Returns:
        List[BaseMessage]: Updated list of conversation messages including the AI response as an AIMessage.

    Raises:
        ValueError: If the input list is empty or contains elements of unknown type.
        RuntimeError: If the call to Azure OpenAI fails for any reason.
    """
    if not messages:
        raise ValueError("Input 'messages' list is empty.")
    if isinstance(messages[0], BaseMessage):
        messages = message_to_dict(messages)
    try:
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model=deployment
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get response from Azure OpenAI: {e}") from e

    content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    base_messages = dict_to_message(messages)
    return base_messages

llm = RunnableLambda(get_response)

if __name__ == "__main__":
    print("running...\n\n")
    messages = [
        SystemMessage(
            content="You are a helpful assistant! Your name is Bob."
        ),
        HumanMessage(
            content="Do fireman only put out fires and get people to the hospital?"
        )
    ]
    output = llm.invoke(messages)
    print(get_buffer_string(output), "\n\n")