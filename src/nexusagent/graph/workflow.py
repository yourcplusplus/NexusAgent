from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nexusagent.graph.nodes import (
    AGENT_STATUS_BAR_NODE,
    agent_status_bar_node,
    agent_status_bar_route,
    chat_responder_node,
    context_compressor_node,
    context_monitor_node,
    context_monitor_route,
    final_node,
    intent_route_fn,
    intent_router_node,
    planner_node,
    verifier_node,
)
from nexusagent.graph.state import NexusGraphState


def build_workflow():
    return build_complex_workflow()


def build_complex_workflow():
    """拓扑:monitor 与 compressor 都先经过 agent_status_bar,再到目标节点。

    这样每次进入 LLM 节点(verifier / 重试的 planner / final)前恰好刷新一次环境
    快照;目标由 status_bar 按 context_next_node 决定,承接旧 context_compressor_route
    的路由职责,压缩与不压缩两条路在此汇合。
    """
    graph = StateGraph(NexusGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("context_monitor", context_monitor_node)
    graph.add_node("context_compressor", context_compressor_node)
    graph.add_node(AGENT_STATUS_BAR_NODE, agent_status_bar_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "context_monitor")
    graph.add_conditional_edges(
        "context_monitor",
        context_monitor_route,
        {"context_compressor": "context_compressor", AGENT_STATUS_BAR_NODE: AGENT_STATUS_BAR_NODE},
    )
    graph.add_edge("context_compressor", AGENT_STATUS_BAR_NODE)
    graph.add_conditional_edges(
        AGENT_STATUS_BAR_NODE,
        agent_status_bar_route,
        {"verifier": "verifier", "planner": "planner", "final": "final"},
    )
    graph.add_edge("verifier", "context_monitor")
    graph.add_edge("final", END)
    return graph.compile()


def build_entry_workflow():
    graph = StateGraph(NexusGraphState)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("chat_responder", chat_responder_node)

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        intent_route_fn,
        {"chat_responder": "chat_responder", "planner": END},
    )
    graph.add_edge("chat_responder", END)
    return graph.compile()
