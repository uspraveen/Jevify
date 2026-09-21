"""Jevify: turn any open LLM into a calibrated System One decision model.

A System One model evaluates a *state* against typed *questions* and returns
probability distributions your code can branch on. Three primitives:

- ``choice``: pick one of K labeled options  -> probabilities over options
- ``score``:  place the state on K ordered levels -> probabilities over levels
- ``noul``:   a yes/no judgment -> a single P(yes)

The wire format is intentionally identical to TypeSafe's Jev API so existing
clients work against a Jevified model by changing a base URL.
"""

__version__ = "0.0.1"

from .load import JevifiedModel, load_jevified  # noqa: E402

__all__ = ["JevifiedModel", "load_jevified", "__version__"]
