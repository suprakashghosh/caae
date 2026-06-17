"""Tests for the CAAE validation pipeline — schema checker, DeepEval gates, and pipeline orchestration."""

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from caae.models.schemas.base import SchemaRegistry
from caae.validation import (
    ValidationPipeline,
    ValidationPipelineResult,
    execute_pydantic_schema_check,
    run_groundedness_assertion,
    run_relevancy_assertion,
    run_schema_adherence_assertion,
)


class TestSchemaChecker:
    """Tests for execute_pydantic_schema_check."""

    def test_valid_payload_against_clinical_schema(self) -> None:
        """Valid AppointmentBookingPayload dict validates True."""
        data = {
            "lead_id": "lead-123",
            "practitioner_id": "dr-smith",
            "preferred_date": "2024-01-15",
            "preferred_time": "10:00",
            "appointment_type": "initial_consultation",
            "notes": "Test notes",
        }
        passed, error = execute_pydantic_schema_check(
            result=data,
            schema_contract="schemas.clinical.AppointmentBookingPayload",
        )
        assert passed is True
        assert error is None

    def test_valid_payload_against_media_schema(self) -> None:
        """Valid QuantitativeIntelPayload dict validates True."""
        data = {
            "topic": "competitor analysis",
            "competitor_names": ["CompA", "CompB"],
            "metrics_requested": ["views", "engagement_rate"],
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-31",
            "depth": "detailed",
        }
        passed, error = execute_pydantic_schema_check(
            result=data,
            schema_contract="schemas.media.QuantitativeIntelPayload",
        )
        assert passed is True
        assert error is None

    def test_invalid_payload_missing_required_field(self) -> None:
        """Dict missing lead_id (required) returns False."""
        data = {
            "practitioner_id": "dr-smith",
            "preferred_date": "2024-01-15",
            "preferred_time": "10:00",
            "appointment_type": "initial_consultation",
        }
        passed, error = execute_pydantic_schema_check(
            result=data,
            schema_contract="schemas.clinical.AppointmentBookingPayload",
        )
        assert passed is False
        assert error is not None

    def test_invalid_payload_extra_fields(self) -> None:
        """Dict with extra fields returns False (extra='forbid')."""
        data = {
            "lead_id": "lead-123",
            "practitioner_id": "dr-smith",
            "preferred_date": "2024-01-15",
            "preferred_time": "10:00",
            "appointment_type": "initial_consultation",
            "extra_field": "should not be here",
        }
        passed, error = execute_pydantic_schema_check(
            result=data,
            schema_contract="schemas.clinical.AppointmentBookingPayload",
        )
        assert passed is False
        assert error is not None

    def test_unknown_schema_contract(self) -> None:
        """Non-existent schema contract returns False."""
        data = {"lead_id": "test"}
        passed, error = execute_pydantic_schema_check(
            result=data,
            schema_contract="schemas.does.not.Exist",
        )
        assert passed is False
        assert error is not None

    def test_custom_registry(self) -> None:
        """Custom SchemaRegistry with a registered schema works correctly."""
        registry = SchemaRegistry()

        class CustomModel(BaseModel):
            name: str
            value: int

        registry.register("schemas.custom.CustomModel", CustomModel)
        data = {"name": "test", "value": 42}
        passed, error = execute_pydantic_schema_check(
            result=data,
            schema_contract="schemas.custom.CustomModel",
            registry=registry,
        )
        assert passed is True
        assert error is None

    def test_empty_result_dict(self) -> None:
        """Empty dict fails validation (missing required fields)."""
        passed, error = execute_pydantic_schema_check(
            result={},
            schema_contract="schemas.clinical.AppointmentBookingPayload",
        )
        assert passed is False
        assert error is not None

    # -- Edge cases ---------------------------------------------------------

    def test_none_result_returns_false(self) -> None:
        """None result returns (False, error)."""
        passed, error = execute_pydantic_schema_check(
            result=None,  # type: ignore[arg-type]
            schema_contract="schemas.clinical.AppointmentBookingPayload",
        )
        assert passed is False
        assert error is not None

    def test_non_validation_error_returns_false(self) -> None:
        """RuntimeError during model_validate returns (False, error)."""

        class FaultyModel(BaseModel):
            name: str

            @classmethod
            def model_validate(cls, data):  # type: ignore[override]
                raise RuntimeError("simulated internal error")

        registry = SchemaRegistry()
        registry.register("schemas.faulty.FaultyModel", FaultyModel)
        passed, error = execute_pydantic_schema_check(
            result={"name": "test"},
            schema_contract="schemas.faulty.FaultyModel",
            registry=registry,
        )
        assert passed is False
        assert error is not None
        assert "simulated internal error" in error


class TestDeepEvalGates:
    """Tests for DeepEval quality gates (all mocked to avoid LLM calls)."""

    # -- Groundedness -------------------------------------------------------

    @patch("deepeval.metrics.FaithfulnessMetric")
    def test_groundedness_pass(self, mock_metric_cls: MagicMock) -> None:
        """Groundedness score >= threshold returns (True, score)."""
        mock_metric = MagicMock()
        mock_metric.score = 0.97
        mock_metric_cls.return_value = mock_metric

        passed, score = run_groundedness_assertion(
            output="some output",
            context="reference context",
        )

        assert passed is True
        assert score == 0.97
        mock_metric.measure.assert_called_once()

    @patch("deepeval.metrics.FaithfulnessMetric")
    def test_groundedness_fail(self, mock_metric_cls: MagicMock) -> None:
        """Groundedness score < threshold returns (False, score)."""
        mock_metric = MagicMock()
        mock_metric.score = 0.80
        mock_metric_cls.return_value = mock_metric

        passed, score = run_groundedness_assertion(
            output="some output",
            context="reference context",
        )

        assert passed is False
        assert score == 0.80

    # -- Relevancy ----------------------------------------------------------

    @patch("deepeval.metrics.AnswerRelevancyMetric")
    def test_relevancy_pass(self, mock_metric_cls: MagicMock) -> None:
        """Relevancy score >= threshold returns (True, score)."""
        mock_metric = MagicMock()
        mock_metric.score = 0.95
        mock_metric_cls.return_value = mock_metric

        passed, score = run_relevancy_assertion(
            output="some output",
            input_="some input",
        )

        assert passed is True
        assert score == 0.95

    @patch("deepeval.metrics.AnswerRelevancyMetric")
    def test_relevancy_fail(self, mock_metric_cls: MagicMock) -> None:
        """Relevancy score < threshold returns (False, score)."""
        mock_metric = MagicMock()
        mock_metric.score = 0.50
        mock_metric_cls.return_value = mock_metric

        passed, score = run_relevancy_assertion(
            output="some output",
            input_="some input",
        )

        assert passed is False
        assert score == 0.50

    # -- Schema Adherence ---------------------------------------------------

    class _DummySchema(BaseModel):
        key: str

    @patch("deepeval.test_case.LLMTestCase")
    @patch("deepeval.metrics.JsonCorrectnessMetric")
    def test_schema_adherence_pass(self, mock_metric_cls: MagicMock, mock_test_case: MagicMock) -> None:
        """Schema adherence score == threshold returns (True, score)."""
        mock_metric = MagicMock()
        mock_metric.score = 1.0
        mock_metric_cls.return_value = mock_metric

        passed, score = run_schema_adherence_assertion(
            output={"key": "value"},
            schema=self._DummySchema,
        )

        assert passed is True
        assert score == 1.0

    @patch("deepeval.test_case.LLMTestCase")
    @patch("deepeval.metrics.JsonCorrectnessMetric")
    def test_schema_adherence_fail(self, mock_metric_cls: MagicMock, mock_test_case: MagicMock) -> None:
        """Schema adherence score < threshold returns (False, score)."""
        mock_metric = MagicMock()
        mock_metric.score = 0.50
        mock_metric_cls.return_value = mock_metric

        passed, score = run_schema_adherence_assertion(
            output={"key": "value"},
            schema=self._DummySchema,
        )

        assert passed is False
        assert score == 0.50

    # -- Import error paths -------------------------------------------------

    @patch.dict("sys.modules", {"deepeval": None})
    def test_groundedness_import_error(self) -> None:
        """ImportError when deepeval not available returns (False, 0.0)."""
        passed, score = run_groundedness_assertion("output", "context")
        assert passed is False
        assert score == 0.0

    @patch.dict("sys.modules", {"deepeval": None})
    def test_relevancy_import_error(self) -> None:
        """ImportError for relevancy returns (False, 0.0)."""
        passed, score = run_relevancy_assertion("output", "input")
        assert passed is False
        assert score == 0.0

    @patch.dict("sys.modules", {"deepeval": None})
    def test_schema_adherence_import_error(self) -> None:
        """ImportError for schema adherence returns (False, 0.0)."""
        passed, score = run_schema_adherence_assertion({"key": "val"}, self._DummySchema)
        assert passed is False
        assert score == 0.0

    # -- Exception during measure -------------------------------------------

    @patch("deepeval.metrics.FaithfulnessMetric")
    def test_groundedness_exception_during_measure(self, mock_metric_cls: MagicMock) -> None:
        """Exception during metric.measure() returns (False, 0.0)."""
        mock_metric = MagicMock()
        mock_metric.measure.side_effect = RuntimeError("LLM API error")
        mock_metric_cls.return_value = mock_metric

        passed, score = run_groundedness_assertion("output", "context")

        assert passed is False
        assert score == 0.0


class _TestSchemaModel(BaseModel):
    """Minimal model for pipeline schema_adherence tests."""

    lead_id: str
    practitioner_id: str
    preferred_date: str
    preferred_time: str
    appointment_type: str


class TestValidationPipeline:
    """Tests for ValidationPipeline orchestration."""

    def test_schema_check_only_pass(self) -> None:
        """Pipeline with skip_deepeval=True and valid data passes."""
        pipeline = ValidationPipeline(
            schema_contract="schemas.clinical.AppointmentBookingPayload",
            skip_deepeval=True,
        )
        data = {
            "lead_id": "lead-123",
            "practitioner_id": "dr-smith",
            "preferred_date": "2024-01-15",
            "preferred_time": "10:00",
            "appointment_type": "initial_consultation",
        }
        result = pipeline.validate(result=data)

        assert result.schema_check_passed is True
        assert result.schema_check_error is None
        assert result.overall_passed is True
        # DeepEval fields remain None when skipped
        assert result.groundedness_passed is None
        assert result.groundedness_score is None
        assert result.relevancy_passed is None
        assert result.relevancy_score is None
        assert result.schema_adherence_passed is None
        assert result.schema_adherence_score is None

    def test_schema_check_only_fail(self) -> None:
        """Pipeline with skip_deepeval=True and invalid data fails schema."""
        pipeline = ValidationPipeline(
            schema_contract="schemas.clinical.AppointmentBookingPayload",
            skip_deepeval=True,
        )
        result = pipeline.validate(result={})  # empty dict fails

        assert result.schema_check_passed is False
        assert result.schema_check_error is not None
        assert result.overall_passed is False
        # DeepEval skipped because schema check failed
        assert result.groundedness_passed is None
        assert result.groundedness_score is None
        assert result.relevancy_passed is None
        assert result.relevancy_score is None
        assert result.schema_adherence_passed is None
        assert result.schema_adherence_score is None

    @patch("caae.validation.run_schema_adherence_assertion")
    @patch("caae.validation.run_relevancy_assertion")
    @patch("caae.validation.run_groundedness_assertion")
    def test_deep_eval_checks_included(
        self,
        mock_groundedness: MagicMock,
        mock_relevancy: MagicMock,
        mock_adherence: MagicMock,
    ) -> None:
        """All DeepEval checks pass when included (skip_deepeval=False)."""
        mock_groundedness.return_value = (True, 0.96)
        mock_relevancy.return_value = (True, 0.92)
        mock_adherence.return_value = (True, 1.0)

        pipeline = ValidationPipeline(
            schema_contract="schemas.clinical.AppointmentBookingPayload",
            skip_deepeval=False,
        )
        data = {
            "lead_id": "lead-123",
            "practitioner_id": "dr-smith",
            "preferred_date": "2024-01-15",
            "preferred_time": "10:00",
            "appointment_type": "initial_consultation",
        }
        result = pipeline.validate(
            result=data,
            context="reference context",
            input_text="user query",
            schema_model=_TestSchemaModel,
        )

        assert result.schema_check_passed is True
        assert result.schema_check_error is None
        assert result.groundedness_passed is True
        assert result.groundedness_score == 0.96
        assert result.relevancy_passed is True
        assert result.relevancy_score == 0.92
        assert result.schema_adherence_passed is True
        assert result.schema_adherence_score == 1.0
        assert result.overall_passed is True

        import json

        expected_text = json.dumps(data, default=str)
        mock_groundedness.assert_called_once_with(
            output=expected_text,
            context="reference context",
            input_text="user query",
        )
        mock_relevancy.assert_called_once_with(
            output=expected_text,
            input_="user query",
        )
        mock_adherence.assert_called_once_with(
            output=data,
            schema=_TestSchemaModel,
            input_text="user query",
        )

    @patch("caae.validation.run_schema_adherence_assertion")
    @patch("caae.validation.run_relevancy_assertion")
    @patch("caae.validation.run_groundedness_assertion")
    def test_schema_fail_skips_deepeval(
        self,
        mock_groundedness: MagicMock,
        mock_relevancy: MagicMock,
        mock_adherence: MagicMock,
    ) -> None:
        """If schema check fails, DeepEval checks are not run."""
        pipeline = ValidationPipeline(
            schema_contract="schemas.clinical.AppointmentBookingPayload",
            skip_deepeval=False,
        )
        result = pipeline.validate(
            result={},  # empty = fails schema
            context="irrelevant",
            input_text="irrelevant",
            schema_model=_TestSchemaModel,
        )

        assert result.schema_check_passed is False
        assert result.schema_check_error is not None
        assert result.overall_passed is False
        assert result.groundedness_passed is None
        assert result.relevancy_passed is None
        assert result.schema_adherence_passed is None

        mock_groundedness.assert_not_called()
        mock_relevancy.assert_not_called()
        mock_adherence.assert_not_called()

    def test_pipeline_result_repr(self) -> None:
        """ValidationPipelineResult.__repr__ produces expected string."""
        result = ValidationPipelineResult()
        result.schema_check_passed = True
        result.groundedness_passed = True
        result.groundedness_score = 0.97
        result.relevancy_passed = False
        result.relevancy_score = 0.50
        result.schema_adherence_passed = True
        result.schema_adherence_score = 1.0
        result.overall_passed = False

        expected = (
            "ValidationPipelineResult(schema=True, groundedness=True, relevancy=False, adherence=True, overall=False)"
        )
        assert repr(result) == expected

    @patch("caae.validation.run_schema_adherence_assertion")
    @patch("caae.validation.run_relevancy_assertion")
    @patch("caae.validation.run_groundedness_assertion")
    def test_deepeval_partial_skip_when_params_missing(
        self,
        mock_groundedness: MagicMock,
        mock_relevancy: MagicMock,
        mock_adherence: MagicMock,
    ) -> None:
        """Only DeepEval checks with provided params are executed."""
        mock_groundedness.return_value = (True, 0.96)

        pipeline = ValidationPipeline(
            schema_contract="schemas.clinical.AppointmentBookingPayload",
            skip_deepeval=False,
        )
        data = {
            "lead_id": "lead-123",
            "practitioner_id": "dr-smith",
            "preferred_date": "2024-01-15",
            "preferred_time": "10:00",
            "appointment_type": "initial_consultation",
        }
        result = pipeline.validate(
            result=data,
            context="only context provided",
            # no input_text, no schema_model
        )

        assert result.schema_check_passed is True
        assert result.schema_check_error is None
        assert result.groundedness_passed is True
        assert result.groundedness_score == 0.96
        assert result.relevancy_passed is None  # skipped
        assert result.relevancy_score is None
        assert result.schema_adherence_passed is None  # skipped
        assert result.schema_adherence_score is None
        assert result.overall_passed is True

        mock_groundedness.assert_called_once()
        mock_relevancy.assert_not_called()
        mock_adherence.assert_not_called()

    def test_pipeline_result_defaults(self) -> None:
        """Fresh ValidationPipelineResult has expected default values."""
        result = ValidationPipelineResult()
        assert result.schema_check_passed is False
        assert result.schema_check_error is None
        assert result.groundedness_passed is None
        assert result.groundedness_score is None
        assert result.relevancy_passed is None
        assert result.relevancy_score is None
        assert result.schema_adherence_passed is None
        assert result.schema_adherence_score is None
        assert result.overall_passed is False
