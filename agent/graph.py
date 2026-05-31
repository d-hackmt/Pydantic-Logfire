from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import planner, explainer, analyst, creator, web_searcher, formatter


def _route_intent(state: AgentState) -> str:
    return state["intent"]



def build_graph():
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("planner",     planner)
    g.add_node("explainer",   explainer)
    g.add_node("analyst",     analyst)
    g.add_node("creator",     creator)
    g.add_node("web_searcher", web_searcher)
    g.add_node("formatter",   formatter)

    # Entry point
    g.add_edge(START, "planner")

    # Planner routes to one of 4 specialists based on classified intent
    g.add_conditional_edges(
        "planner",
        _route_intent,
        {
            "explain": "explainer",
            "analyze": "analyst",
            "create":  "creator",
            "search":  "web_searcher",
        },
    )

    # All specialists flow into the formatter, then done
    for specialist in ("explainer", "analyst", "creator", "web_searcher"):
        g.add_edge(specialist, "formatter")

    g.add_edge("formatter", END)

    return g.compile()
