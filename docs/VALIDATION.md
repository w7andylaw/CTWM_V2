# CT-WM v13 validation

Revision `ctm-identity-gated-retention-v13`, 2026-10-23.

59 CPU tests passed (zero failures/errors): NRSM numerical and history/gradient
contracts, discrete/continuous reward action alignment, world-model/actor/critic
joint updates, vector/image observation routes, compiled execution, checkpoint
round-trip and old-schema rejection. No GPU training was started by validation.

The production NRSM executable change is exactly one activity assignment,
confirmed by AST comparison after removing docstrings. Reward head, agent/loss,
config and distribution implementation are byte-identical to v12. Existing
parameters/shapes are retained; the schema changes because dynamics changed.

The test includes g=0 activity identity and its gradient, fully open calibrated
decay, partial-gate geometric writing, raw activity access, finite long-T write
coefficients, and an independent NumPy oracle for all tick calculations. The
two transition paths and reset semantics share the tested recurrence.

TeX source compiled successfully in the desktop compiler.75 labels remain;
71 expressions and the native task/reward section are identical to v12.
Only activity update and three derived identities change. The historical disk
PDF is not evidence of the new source; use the editor's compiled preview.

Raw evidence is in [the versioned validation folder](validation/identity_retention_20261023/validation.json)
and [test log](validation/identity_retention_20261023/tests.log).
The source hash/deployment manifest and production compiler receipt are saved
there too. Earlier reports remain associated with their original versions.

The completed same-checkpoint offline control found a small retention trend
but no useful correct/wrong reward ranking. No new online policy, full-batch
GPU benchmark or convergence result is claimed. See
[memory and reward review](MEMORY_AND_REWARD_REVIEW.md).
