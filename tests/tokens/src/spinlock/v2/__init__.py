"""V2 package - DEPRECATED.

This package has been reorganized. V2 components have been moved to their
primary locations.
"""

import warnings

warnings.warn(
    "The 'spinlock.v2' package is deprecated. "
    "V2 components have moved to primary locations:\n"
    "  - spinlock.v2.tokens → spinlock.tokens\n"
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2
)
