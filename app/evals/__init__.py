"""Automated evaluation for the multi-agent platform.

Three layers, all driven by the same case format (`EvalCase`):

  1. Deterministic checks (`metrics`): tool trajectory, policy outcome,
     transfers, answer content, budgets. These are the hard gates -- cheap,
     exact, never flaky.
  2. LLM-as-judge (`judges`): rubric scores for the things code cannot check
     (is the answer grounded in the tool outputs? did it complete the task?).
     Reported as trends and a soft gate, because a judge is itself a model.
  3. Regression (`regression`): a run is compared with the baseline run of
     the same set; a case that flipped from pass to fail, or a metric that
     dropped past a threshold, fails the gate.

Targets: one spec agent (individual) or the router over every approved
spec (end to end: routing + sub-agent + tools + policy). The planner is
evaluated through `generator` / `planner_eval` (intent -> plan quality).
"""
