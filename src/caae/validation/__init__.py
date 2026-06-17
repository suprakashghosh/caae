"""Validation and quality gates."""

import json
from typing import Any

from pydantic import BaseModel

from caae.validation.deepeval_gates import (
    run_groundedness_assertion,
    run_relevancy_assertion,
    run_schema_adherence_assertion,
)
from caae.validation.schema_checker import execute_pydantic_schema_check


class ValidationPipelineResult:
    """Result of a validation pipeline run."""

    def __init__(self) -> None:
        self.schema_check_passed: bool = False
        self.schema_check_error: str | None = None
        self.groundedness_passed: bool | None = None  # None if skipped
        self.groundedness_score: float | None = None
        self.relevancy_passed: bool | None = None  # None if skipped
        self.relevancy_score: float | None = None
        self.schema_adherence_passed: bool | None = None  # None if skipped
        self.schema_adherence_score: float | None = None
        self.overall_passed: bool = False

    def __repr__(self) -> str:
        return (
            f"ValidationPipelineResult("
            f"schema={self.schema_check_passed}, "
            f"groundedness={self.groundedness_passed}, "
            f"relevancy={self.relevancy_passed}, "
            f"adherence={self.schema_adherence_passed}, "
            f"overall={self.overall_passed})"
        )


class ValidationPipeline:
    """Orchestrates all validation checks: schema, groundedness, relevancy, adherence.

    Runs checks in order:
    1. Pydantic schema check (cheap, deterministic)
    2. DeepEval groundedness (requires LLM)
    3. DeepEval relevancy (requires LLM)
    4. DeepEval schema adherence (requires LLM)

    If the schema check fails, subsequent checks are skipped.
    """

    def __init__(
        self,
        schema_contract: str,
        skip_deepeval: bool = False,
    ) -> None:
        """Initialize the pipeline.

        Args:
            schema_contract: Dotted path to the schema contract.
            skip_deepeval: If True, skip DeepEval checks (useful in testing).
        """
        self.schema_contract = schema_contract
        self.skip_deepeval = skip_deepeval

    def validate(
        self,
        result: dict[str, Any],
        context: str | None = None,
        input_text: str | None = None,
        schema_model: type[BaseModel] | None = None,
        output_text: str | None = None,
    ) -> ValidationPipelineResult:
        """Run all validation checks.

        Args:
            result: The output data dict to validate.
            context: Reference context for groundedness check.
            input_text: Original input for relevancy check.
            schema_model: Expected Pydantic BaseModel class for adherence check (optional).
            output_text: Pre-serialized output text for LLM-based metrics (optional).
                Falls back to json.dumps(result, default=str) if not provided.

        Returns:
            ValidationPipelineResult with detailed pass/fail for each check.
        """
        pipeline_result = ValidationPipelineResult()

        # Step 1: Pydantic schema check (deterministic, cheap)
        schema_passed, schema_error = execute_pydantic_schema_check(
            result=result,
            schema_contract=self.schema_contract,
        )
        pipeline_result.schema_check_passed = schema_passed
        pipeline_result.schema_check_error = schema_error

        if not pipeline_result.schema_check_passed:
            pipeline_result.overall_passed = False
            return pipeline_result

        # Steps 2-4: DeepEval checks (optional, can be skipped)
        if self.skip_deepeval:
            pipeline_result.overall_passed = True
            return pipeline_result

        # Common: serialize result once for text-based metrics
        text_for_metric = output_text if output_text is not None else json.dumps(result, default=str)

        # Groundedness
        if context is not None:
            passed, score = run_groundedness_assertion(
                output=text_for_metric,
                context=context,
                input_text=input_text,
            )
            pipeline_result.groundedness_passed = passed
            pipeline_result.groundedness_score = score

        # Relevancy
        if input_text is not None:
            passed, score = run_relevancy_assertion(
                output=text_for_metric,
                input_=input_text,
            )
            pipeline_result.relevancy_passed = passed
            pipeline_result.relevancy_score = score

        # Schema adherence
        if schema_model is not None:
            passed, score = run_schema_adherence_assertion(
                output=result,
                schema=schema_model,
                input_text=input_text,
            )
            pipeline_result.schema_adherence_passed = passed
            pipeline_result.schema_adherence_score = score

        # Overall: schema check passed AND all executed DeepEval checks passed
        all_checks = [pipeline_result.schema_check_passed]
        if pipeline_result.groundedness_passed is not None:
            all_checks.append(pipeline_result.groundedness_passed)
        if pipeline_result.relevancy_passed is not None:
            all_checks.append(pipeline_result.relevancy_passed)
        if pipeline_result.schema_adherence_passed is not None:
            all_checks.append(pipeline_result.schema_adherence_passed)

        pipeline_result.overall_passed = all(all_checks)
        return pipeline_result


__all__ = [
    "execute_pydantic_schema_check",
    "run_groundedness_assertion",
    "run_relevancy_assertion",
    "run_schema_adherence_assertion",
    "ValidationPipelineResult",
    "ValidationPipeline",
]
