"""Machine-written narrative explaining a run.

Three layers, deliberately separated:

* :mod:`evalshift.insights.facts` renders every figure the narrative may
  mention into a display string, so the generating model copies rather than
  calculates.
* :mod:`evalshift.insights.templates` turns those same figures into
  deterministic prose — what ships when generation is skipped or fails.
* :mod:`evalshift.insights.generator` prompts a model and validates its
  output back against the facts.

Only the first two are pure; that is what makes the third testable without
mocking a provider.
"""

from __future__ import annotations
