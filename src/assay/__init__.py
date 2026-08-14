"""assay — stdlib-only capability prober for locally-served LLM endpoints."""

# __version__ is defined before the imports below: assay.run reads it
# from this partially-initialized module during its own import.
__version__ = "0.5.0"

from assay.budget import Budget
from assay.profile import Profile
from assay.run import probe

__all__ = ["Budget", "Profile", "probe", "__version__"]
