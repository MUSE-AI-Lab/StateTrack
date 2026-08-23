# Multi-Constraint Micro-Audit Summary

## Scope
- Real API audit with two simultaneous persistent constraints per dialogue.
- Models attempted: 7; scenarios: 5; valid outputs: 34.
- One GPT-5.4 persona+logical probe hit a repeated upstream 502 and is excluded from valid-output denominators.
- Incremental logged cost: $0.1765 (cap $1.00).

## Aggregate results
- Constraint 1 success: 82.4%
- Constraint 2 success: 82.4%
- Joint two-constraint success: 76.5%
- Forked recall success: 73.5%
- SIC+: 5.9%
- SIC-: 8.8%

## Interpretation for rebuttal
- This supports a calibrated response to the single-constraint critique: multi-constraint interference is indeed harder and should be framed as future/full-benchmark work.
- The result does not replace the primary benchmark; it is a compact diagnostic scope check.
- The response should say the current benchmark intentionally isolates state integrity, and the new micro-audit confirms that adding simultaneous constraints increases difficulty while preserving model-dependent behavior.
