from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages


llm = init_chat_model(
    'qwen3:8b',
    model_provider='ollama'
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def logical_agent(state: AgentState):
    print("logical agent called")
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant who only focus on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}