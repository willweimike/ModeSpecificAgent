from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from agents.emotional_agent import emotional_agent
from agents.logical_agent import logical_agent


llm = init_chat_model(
    'qwen3:8b',
    model_provider='ollama'
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional or logical response."
    )


def classify_message(state: AgentState):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}


def router(state: AgentState):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "emotional"}

    return {"next": "logical"}


graph_builder = StateGraph(AgentState)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("emotional", emotional_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"emotional": "emotional", "logical": "logical"}
)

graph_builder.add_edge("emotional", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()


def main():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input.lower() == "quit":
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    main()
