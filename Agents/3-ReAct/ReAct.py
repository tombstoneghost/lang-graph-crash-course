# Imports
from typing import Annotated, Sequence, TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Annotated - Provides additional context without effecting the data type
# Sequence - Automatically handle the state updates for sequences such as by adding new messages to a chat history


# Load .env file
load_dotenv(dotenv_path="../.env")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseException], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds two numbers together"""

    return a + b


@tool
def subtract(a: int, b: int):
    """This is an subtraction function that find the difference bwteen two numbers together"""

    return a - b


@tool
def multiply(a: int, b: int):
    """This is an multiplication function that multiplies two numbers together"""

    return a * b


tools = [add, subtract, multiply]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state['messages'])

    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "end": END,
        "continue": "tools"
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]

        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Add 3 + 4. Add 34 + 21. Add 12 + 12")]}
print_stream(app.stream(inputs, stream_mode="values"))

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6")]}
print_stream(app.stream(inputs, stream_mode="values"))

