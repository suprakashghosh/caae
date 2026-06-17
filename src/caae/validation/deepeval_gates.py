"""DeepEval quality gates — evaluation metrics for LLM outputs."""

import json
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def run_groundedness_assertion(
    output: str,
    context: str,
    threshold: float = 0.95,
    input_text: str | None = None,
) -> tuple[bool, float]:
    """Run DeepEval FaithfulnessMetric against threshold.

    Args:
        output: The LLM output text.
        context: The reference context to check groundedness against.
        threshold: Minimum score to pass (default 0.95).
        input_text: Optional original input text for the test case.

    Returns:
        Tuple of (passed: bool, score: float).
    """
    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=input_text or "",
            actual_output=output,
            retrieval_context=[context],
        )
        metric = FaithfulnessMetric(threshold=threshold)
        metric.measure(test_case)
        score = metric.score
        return (score >= threshold, score)
    except ImportError:
        logger.warning(
            "deepeval not installed — skipping groundedness assertion",
        )
        return (False, 0.0)
    except Exception as exc:
        logger.error(
            "DeepEval groundedness assertion failed: %s",
            exc,
        )
        return (False, 0.0)


def run_relevancy_assertion(
    output: str,
    input_: str,
    threshold: float = 0.90,
) -> tuple[bool, float]:
    """Run DeepEval AnswerRelevancyMetric against threshold.

    Args:
        output: The LLM output text.
        input_: The original input/prompt.
        threshold: Minimum score to pass (default 0.90).

    Returns:
        Tuple of (passed: bool, score: float).
    """
    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=input_,
            actual_output=output,
        )
        metric = AnswerRelevancyMetric(threshold=threshold)
        metric.measure(test_case)
        score = metric.score
        return (score >= threshold, score)
    except ImportError:
        logger.warning(
            "deepeval not installed — skipping relevancy assertion",
        )
        return (False, 0.0)
    except Exception as exc:
        logger.error(
            "DeepEval relevancy assertion failed: %s",
            exc,
        )
        return (False, 0.0)


def run_schema_adherence_assertion(
    output: dict[str, Any],
    schema: type[BaseModel],
    threshold: float = 1.0,
    input_text: str | None = None,
) -> tuple[bool, float]:
    """Run DeepEval JsonCorrectnessMetric against threshold.

    Args:
        output: The LLM output as a dict.
        schema: The expected Pydantic BaseModel class.
        threshold: Minimum score to pass (default 1.0 = 100%).
        input_text: Optional original input text for the test case.

    Returns:
        Tuple of (passed: bool, score: float).
    """
    try:
        from deepeval.metrics import JsonCorrectnessMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=input_text or "",
            actual_output=json.dumps(output, default=str),
        )
        metric = JsonCorrectnessMetric(
            expected_schema=schema,
            threshold=threshold,
        )
        metric.measure(test_case)
        score = metric.score
        return (score >= threshold, score)
    except ImportError:
        logger.warning(
            "deepeval not installed — skipping schema adherence assertion",
        )
        return (False, 0.0)
    except Exception as exc:
        logger.error(
            "DeepEval schema adherence assertion failed: %s",
            exc,
        )
        return (False, 0.0)
