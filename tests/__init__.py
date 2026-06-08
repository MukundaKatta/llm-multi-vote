"""Test package for llm-multi-vote.

Ensures the ``src/`` layout package is importable when the suite is run via
``python3 -m unittest discover -s tests`` from the repository root without an
editable install. An installed package already on ``sys.path`` takes
precedence, so this is a no-op in CI where ``pip install -e`` is used.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
