"""Evaluation gate node — validates outputs and orchestrates conditional routing."""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

from caae.models.config import IntentRoute
from caae.models.state import UnifiedContextState
from caae.nodes.deps import Dependencies

logger = logging.getLogger(__name__)


def _get_deps(config: RunnableConfig) -> Dependencies:
    """Extract the Dependencies container from the LangGraph config."""
    return config["configurable"]["deps"]


async def evaluation_gate_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Validate execution results against the registered schema contract.

    Steps:
        1. If ``validation_retry_count`` is already >= 3, set ``evaluation_passed = False``
           and return unchanged count (routing function handles escalation).
        2. Resolve the schema contract from the workflow policy using ``resolved_intent``.
        3. Validate ``execution_mutation_result`` (and fall back to ``extracted_quantitative_data``)
           against the resolved Pydantic schema.
        4. Also verify ``resolved_intent`` is not ``"unknown"``.
        5. If all checks pass: ``evaluation_passed = True``, retry count unchanged.
        6. If any check fails: ``evaluation_passed = False``, retry count incremented (capped at 3).

    Returns:
        ``{"validation_retry_count": int, "evaluation_passed": bool | None}``
    """
    # LangGraph may pass state as a Pydantic model; normalize to dict
    if not isinstance(state, dict):
        state = state.model_dump()

    deps = _get_deps(config)
    retry_count: int = state.get("validation_retry_count", 0)
    resolved_intent: str | None = state.get("resolved_intent")

    # ── Already exhausted retries ───────────────────────────────────────────
    if retry_count >= 3:
        logger.warning("Validation retry count exhausted (%d/3); gate not passed", retry_count)
        return {
            "validation_retry_count": retry_count,
            "evaluation_passed": False,
        }

    evaluation_passed: bool = True
    new_retry_count: int = retry_count  # unchanged unless validation fails

    # ── Validate intent ─────────────────────────────────────────────────────
    if not resolved_intent or resolved_intent == "unknown":
        logger.warning("Validation failed: resolved_intent is '%s'", resolved_intent)
        evaluation_passed = False

    # ── Validate execution result status ───────────────────────────────────
    execution_result = state.get("execution_mutation_result", {})
    if isinstance(execution_result, dict) and execution_result.get("status") not in ("completed", None, ""):
        logger.warning(
            "Validation failed: execution_mutation_result status is '%s'",
            execution_result.get("status"),
        )
        evaluation_passed = False

    # ── Validate execution result / extracted data against schema ───────────
    if evaluation_passed:
        route: IntentRoute | None = deps.workflow_policy.intent_routing_matrix.get(resolved_intent)  # type: ignore[arg-type]
        if route is None:
            logger.warning("Validation failed: no route for intent '%s'", resolved_intent)
            evaluation_passed = False
        else:
            evaluation_passed = _validate_against_schema(
                deps=deps,
                route=route,
                state=state,
            )

    # ── Update retry count ──────────────────────────────────────────────────
    if not evaluation_passed:
        new_retry_count = min(retry_count + 1, 3)

    return {
        "validation_retry_count": new_retry_count,
        "evaluation_passed": evaluation_passed,
    }


def _validate_against_schema(
    deps: Dependencies,
    route: IntentRoute,
    state: dict[str, Any],
) -> bool:
    """Validate execution results and extracted data against the schema contract.

    Tries ``execution_mutation_result`` first, then falls back to
    ``extracted_quantitative_data``.
    """
    try:
        schema_class: type[BaseModel] = deps.schema_registry.resolve(route.runtime_schema_contract)
    except Exception:
        logger.exception("Schema contract '%s' not found in registry", route.runtime_schema_contract)
        return False

    # Prefer extracted_quantitative_data (structured LLM output matching schema),
    # fall back to execution_mutation_result.
    target_data: dict[str, Any] | None = state.get("extracted_quantitative_data")
    if not target_data or not isinstance(target_data, dict):
        target_data = state.get("execution_mutation_result")
    if not target_data or not isinstance(target_data, dict):
        logger.warning("No data available to validate against schema '%s'", route.runtime_schema_contract)
        return False

    # Check for explicit error markers from upstream nodes
    if target_data.get("error") or target_data.get("status") == "skipped":
        logger.warning("Validation rejected: data contains error/status=skipped")
        return False

    try:
        schema_class(**target_data)
        logger.info(
            "Validation passed against schema '%s'",
            route.runtime_schema_contract,
        )
        return True
    except ValidationError as exc:
        logger.warning(
            "Validation failed against schema '%s': %s",
            route.runtime_schema_contract,
            exc,
        )
        return False
    except Exception:
        logger.exception("Unexpected validation error")
        return False


def verify_output_compliance(state: UnifiedContextState) -> str:
    """Determine the next routing edge after the evaluation gate.

    Called by LangGraph's ``add_conditional_edges`` as the routing function.
    Reads the **already-updated** state to decide the next node.

    Returns:
        - ``"human_handoff_escalation"`` if retries exhausted (>= 3)
        - ``"commit_state_and_exit"`` if ``evaluation_passed`` is ``True``
        - ``"re_evaluate_context_node"`` otherwise
    """
    # Normalize state for safety (LangGraph may pass dict or Pydantic model)
    if isinstance(state, dict):
        retry_count = state.get("validation_retry_count", 0)
        passed = state.get("evaluation_passed")
    else:
        retry_count = state.validation_retry_count
        passed = state.evaluation_passed

    if retry_count >= 3:
        logger.warning("Retry limit reached (%d/3); escalating to human handoff", retry_count)
        return "human_handoff_escalation"

    if passed is True:
        logger.info("Evaluation passed; committing state and exiting")
        return "commit_state_and_exit"

    logger.info("Evaluation not passed; re-evaluating context")
    return "re_evaluate_context_node"
