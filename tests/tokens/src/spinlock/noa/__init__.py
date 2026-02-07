"""NOA package - DEPRECATED.

The NOA package has been renamed to MNO (Meta-Neural Operator).

DEPRECATED: Import from spinlock.mno instead of spinlock.noa
"""

import warnings

warnings.warn(
    "spinlock.noa is deprecated. "
    "The package has been renamed to spinlock.mno:\n"
    "  Old: from spinlock.noa import NOABackbone\n"
    "  New: from spinlock.mno import NOABackbone\n"
    "\n"
    "This deprecation shim will be removed in 6 months (August 2026).",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from mno for backward compatibility
from spinlock.mno import *  # noqa: F401,F403

__all__ = [
    "NOABackbone",
    "BaseNOABackbone",
]
