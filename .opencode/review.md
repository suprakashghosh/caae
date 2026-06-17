# Code Review Summary

**Scope**: Sub-Task 6 — Validation Pipeline (Pydantic + DeepEval): `src/caae/validation/schema_checker.py`, `src/caae/validation/deepeval_gates.py`, `src/caae/validation/__init__.py`, `tests/unit/test_validation.py`.

**Overall risk**: High

**Verdict**: Request changes

## Findings

### [P0] Blocking

- **DeepEval gate imports reference classes/submodules that do not exist in the locked `deepeval==4.0.6`**
  - **Location**: `src/caae/validation/deepeval_gates.py:25-26,66-69,108-109`
  - **Why it matters**: In the current environment (`uv.lock` pins `deepeval==4.0.6`) all three lazy imports fail with `ModuleNotFoundError`: `deepeval.metrics.groundedness`, `deepeval.metrics.conversational_relevancy`, and `deepeval.metrics.schema_adherence` are gone. The fallback `except ImportError` returns `(False, 0.0)` with a warning, so the gates silently never pass in production.
  - **Evidence**: `uv run python -c "from deepeval.metrics.groundedness import GroundednessMetric"` raises `ModuleNotFoundError: No module named 'deepeval.metrics.groundedness'`. Same for the other two. Available 4.0.6 equivalents are `FaithfulnessMetric`, `ContextualRelevancyMetric`/`AnswerRelevancyMetric`, and `JsonCorrectnessMetric`. Tests only pass because `tests/unit/test_validation.py` injects fake modules for the missing submodules.
  - **Fix**: Either (a) pin `deepeval` to the last version that exposes the old names in `pyproject.toml` and regenerate `uv.lock`, or (b) migrate the gates to the 4.x API. If migrating, note `JsonCorrectnessMetric` expects `expected_schema: pydantic.BaseModel`, not a raw dict, and its `actual_output` is a string; adjust `run_schema_adherence_assertion` signature/behavior accordingly.

### [P1] High

- **`ValidationPipeline.validate` passes `str(result)` to text metrics for structured payloads**
  - **Location**: `src/caae/validation/__init__.py:99,108`
  - **Why it matters**: Groundedness and relevancy metrics score text. Passing Python's `str({'lead_id': 'lead-123', ...})` instead of the actual LLM-generated string or JSON gives the metric a_repr_ string it was not designed to evaluate, producing unreliable pass/fail decisions.
  - **Evidence**: `pipeline.validate(result=data, context=...)` calls `run_groundedness_assertion(output=str(data), context=...)`.
  - **Fix**: Add an optional `output_text: str | None = None` parameter to `validate`. Use it when provided; otherwise serialize `result` with `json.dumps(result, default=str)` rather than `str(result)`.

### [P2] Medium

- **`ValidationPipelineResult` omits failure reasons from the "detailed breakdown"**
  - **Location**: `src/caae/validation/__init__.py:11-32`
  - **Why it matters**: The acceptance criteria requires a "clear pass/fail verdict with detailed breakdown". The result exposes booleans and scores, but callers cannot tell *why* the schema check failed (missing field, extra field, unknown contract, or unexpected error) without re-reading logs.
  - **Evidence**: `execute_pydantic_schema_check` logs warnings/errors but returns only `bool`.
  - **Fix**: Add `schema_check_error: str | None = None` to `ValidationPipelineResult` and have `execute_pydantic_schema_check` return error detail (e.g., change return type to `tuple[bool, str | None]` or raise a custom exception). Update tests.

- **Missing unit tests for explicit edge cases**
  - **Location**: `tests/unit/test_validation.py`
  - **Why it matters**: The review criteria list empty dicts, unknown contracts, and `None` inputs. Empty dict and unknown contract are covered; `None` input and non-`ValidationError` exceptions during `model_validate` are not.
  - **Evidence**: No `test_none_result` or test where `model_validate` raises `RuntimeError`/`AttributeError`.
  - **Fix**: Add tests asserting `execute_pydantic_schema_check(None, "schemas.clinical.AppointmentBookingPayload") is False` and that arbitrary exceptions during validation return `False`.

### [P3] Low

- **Type annotation inconsistency in `ValidationPipeline.validate`**
  - **Location**: `src/caae/validation/__init__.py:61-66`
  - **Why it matters**: `result: dict` and `schema_dict: dict | None` lack type parameters; the rest of the package uses `dict[str, Any]`.
  - **Fix**: Use `dict[str, Any] | None` for `result`, `schema_dict`, and `input_text` parameters for consistency.

- **`LLMTestCase` constructed with empty `input=""` for groundedness and schema adherence**
  - **Location**: `src/caae/validation/deepeval_gates.py:29,112`
  - **Why it matters**: DeepEval's docs state `input` and `actual_output` are mandatory; while the metrics may tolerate empty input, providing the original user query when available yields more accurate evaluation.
  - **Fix**: Add `input_text: str | None` parameters to `run_groundedness_assertion` and `run_schema_adherence_assertion`, and pass `input_text or ""` through the pipeline.

- **Plan signature mismatch for gate return types**
  - **Location**: `.opencode/plan.md:397-399` vs `src/caae/validation/deepeval_gates.py:13-14,54-55,96-97`
  - **Why it matters**: The plan states gates return `bool`; implementation returns `tuple[bool, float]`. The acceptance criteria explicitly asks for scores, so the implementation is preferable, but the plan should be updated to avoid future confusion.
  - **Fix**: Update `.opencode/plan.md` to document `tuple[bool, float]` return types.

## Positive Notes

- `execute_pydantic_schema_check` correctly delegates to `SchemaRegistry.resolve()` and handles `SchemaRegistryError`, `ValidationError`, and unexpected exceptions gracefully.
- `ValidationPipeline` correctly skips all DeepEval checks when the schema check fails and only runs gates whose parameters are supplied.
- Lazy imports inside gate functions keep DeepEval optional and avoid hard dependency failures at module load time.
- All 24 unit tests pass (`uv run pytest tests/unit/test_validation.py -v`).

## Suggested Next Steps

- [ ] Fix DeepEval metric imports/API mismatch before merge (P0).
- [ ] Decide whether `ValidationPipeline.validate` should receive raw LLM output text or structured result serialization (P1).
- [ ] Add failure reasons to `ValidationPipelineResult` and propagate schema errors (P2).
- [ ] Add unit tests for `None` input and non-validation exceptions in schema checker (P2).
- [ ] Align type annotations and pass `input_text` to all relevant gates (P3).
- [ ] Update `.opencode/plan.md` gate return-type signatures (P3).
- [ ] Re-run `pytest tests/unit/test_validation.py` and, if possible, a real DeepEval smoke test after changes.
