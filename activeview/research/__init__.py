"""Local-first, auditable research experiment infrastructure."""

from .experiment import Decision, Experiment, ExperimentStatus
from .test_gate import TestGateError, assert_test_allowed, validate_final_test_authorization

__all__ = [
    "Decision",
    "Experiment",
    "ExperimentStatus",
    "TestGateError",
    "assert_test_allowed",
    "validate_final_test_authorization",
]
