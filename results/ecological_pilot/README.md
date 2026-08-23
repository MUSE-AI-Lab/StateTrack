# SGD dialogue pilot

This directory contains a 30-case analysis of persistent constraints in public human-authored task-oriented dialogues from the Schema-Guided Dialogue dataset. It is independent of the submitted StateTrack-Bench tables.

The case set contains 15 carryover cases, 10 overwrite cases, and 5 revocation or relaxation cases. Each case was evaluated with seven models. Every model-case record contains a behavior response and a separately elicited recall response, giving 210 judged records and 420 subject-model calls in total.

## Results

| Measure | Count | Rate |
|---|---:|---:|
| Active-constraint compliance | 187/210 | 89.0% |
| Strict joint success | 140/210 | 66.7% |
| Recall | 78/210 | 37.1% |
| SIC+ | 9/210 | 4.3% |
| SIC- | 118/210 | 56.2% |

For the five revocation or relaxation cases, active-constraint compliance and recall are both 34/35. The complete model-level and case-type breakdown, including Wilson 95% confidence intervals, is stored in `summary.json`.

## Files

- `selected_cases.json`: dialogue context, active state, target turn, and case type for the 30 selected cases.
- `case_audit.csv`: compact manual audit sheet for the selected cases.
- `raw_subject_outputs.json`: behavior and recall responses from the seven models.
- `judged_outputs.json`: task, compliance, recall, SIC, and old-rule labels with short rationales.
- `summary.json` and `summary.md`: aggregate results.
- `sgd_manifest.json`: source files and hashes for the SGD snapshot used in case selection.
- `checksums.json`: hashes for the pilot data files.

## Reproduce the summary

Run these commands from the repository root:

```bash
python scripts/sgd_ecological_pilot/reproduce_pilot_metrics.py
python scripts/sgd_ecological_pilot/verify_checksums.py
```

The first command recomputes `summary.json` from `judged_outputs.json`. The second verifies the pilot data hashes.
