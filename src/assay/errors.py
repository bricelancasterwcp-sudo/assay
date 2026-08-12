"""assay error taxonomy (spec §3).

Infrastructure failures raise; model behavior (refusal, rambling,
wrong format) is DATA, never an exception — there is no ModelError.
"""


class AssayError(Exception):
    """Base class for all assay errors."""


class InfrastructureError(AssayError):
    """The endpoint failed us: transport error, HTTP 5xx, timeout."""


class ContractViolation(InfrastructureError):
    """A 200 response that breaks the response contract."""


class BudgetExhausted(AssayError):
    """A charge would cross a budget limit; nothing was recorded."""
