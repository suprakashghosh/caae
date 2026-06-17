"""LangGraph builder — constructs the state-graph pipeline for CAAE."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from caae.models.state import UnifiedContextState
from caae.nodes import (
    action_execution_node,
    cognitive_processing_node,
    context_assessor_node,
    evaluation_gate_node,
    info_retrieval_node,
    verify_output_compliance,
)


def build_caae_graph() -> CompiledStateGraph:
    """Build and compile the CAAE LangGraph.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(UnifiedContextState)

    # ── Register nodes ──────────────────────────────────────────────────────
    graph.add_node("context_assessor", context_assessor_node)
    graph.add_node("info_retrieval", info_retrieval_node)
    graph.add_node("cognitive_processing", cognitive_processing_node)
    graph.add_node("action_execution", action_execution_node)
    graph.add_node("evaluation_gate", evaluation_gate_node)

    # ── Linear forward edges ─────────────────────────────────────────────────
    graph.add_edge(START, "context_assessor")
    graph.add_edge("context_assessor", "info_retrieval")
    graph.add_edge("info_retrieval", "cognitive_processing")
    graph.add_edge("cognitive_processing", "action_execution")
    graph.add_edge("action_execution", "evaluation_gate")

    # ── Conditional routing from evaluation gate ─────────────────────────────
    graph.add_conditional_edges(
        "evaluation_gate",
        verify_output_compliance,
        {
            "commit_state_and_exit": END,
            "re_evaluate_context_node": "context_assessor",
            "human_handoff_escalation": END,
        },
    )

    compiled: CompiledStateGraph = graph.compile()
    return compiled
