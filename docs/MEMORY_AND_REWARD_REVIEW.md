# v13: memory preservation and usable reward output

2026-10-23. This review separates published mechanisms, inspected code, measured
offline results, and proposed follow-up controls. The production change is only
the identity-gated activity recurrence. No further module, loss, or training
intervention described below has been enabled.

## What is being preserved?

For Memory Length, the relevant requirement is to retain the initial cue until
the terminal decision, in a form the reward model can use together with the
generating action. Preserving every observation is unnecessary. Nonzero activity,
large RMS, or a long time constant does not establish cue preservation. Probe
accuracy measures held-out decodability under a specified probe; it is not mutual
information, a fraction of all stored information, or policy success.

Three checks are distinct: (1) information enters the recurrent state; (2) it
survives distractors and internal ticks; (3) the existing readout uses it to make
different predictions for the two actions. Stable coordinates also matter: a
probe refitted at every time can hide changes in how the cue is represented.

## SSM / R2I

A single SSM layer has x_t=Abar*x_(t-1)+Bbar*u_t, y_t=C*x_t+D*u_t.
For two histories with identical subsequent inputs, their state difference is
Abar^L*delta_x. This conditional algebra isolates a direct linear retention path
from writing and output processing. Slow modes can keep a cue, but contraction,
interference and a poor readout can still destroy its usefulness. It is not an
exact archive of arbitrary input history. R2I combines diagonal MIMO SSMs with
HiPPO-derived initialization, learned discretization timescales, nonlinear output
processing, a non-recurrent observation posterior and long training sequences.
These design choices form a trainable route from observations to predictions.
Its reward head reads output h and stochastic z; direct hidden x access is a
separate policy-input design, not a guarantee that the reward head reads x.
See [R2I §§2–3 and Appendices B, M–O](https://arxiv.org/html/2403.04253v1).

Inspected code: recall2imagine/ssm/mimo.py (discretization/HiPPO parameters),
ssm/common.py (state recurrence and output block), ssm_nets.py, configs.yaml.
The linear stored state is returned separately from the GELU/GLU/residual/norm
output. The retained project default has clip_eigs=False, so do not claim an
enforced stable spectrum throughout training. The per-layer difference identity
above is not a global theorem for the complete nonlinear S3M stack or imagination.

## CTM

Original CTM uses private neuron models over recent pre-activation histories and
pairwise temporal synchronization. The RL implementation carries pre/post traces
between environment steps and uses a finite synchronization window. It does not
have the v13 gated identity activity path. In the inspected implementation,
current observation features enter every tick directly, unlike CTWM's observation
information arriving through sampled z on the next transition. A short FIFO can
retain older information through recurrence only if the dynamics learn to keep it.
Synchronization reads available history; it cannot reconstruct a cue already lost.
See [CTM Appendix I.2](https://arxiv.org/html/2505.05522v1#A9.SS2) and
[official RL implementation](https://github.com/SakanaAI/continuous-thought-machines/blob/main/models/ctm_rl.py).

Static maze reasoning repeatedly queries features of an available map; it is not
the same information-access problem as a one-time cue followed by distractors.
Our raw activity path also avoids forcing all information through pair products:
a global sign flip leaves pairwise products unchanged, while raw activity changes.
That algebra explains the usefulness of keeping activity, but does not prove the
actual learned cue is encoded as a global sign flip.

## CMM

CMM adds persistent LTM tokens beside the rolling STM and jointly reads/writes
them with a Transformer. The updated LTM persists; the transformed STM is used
by NLMs and then discarded, while the original STM continues as the rolling
window. In Maze/Parity, residual blocks have a learned scalar initialized to0;
other experiments use a fixed1. Thus only those blocks start as identity maps.
LTM is additionally normalized before carry, so the entire memory update is not
automatically identity. CMM separates storage from processing; this is the useful
principle here, not evidence that more internal computation protects memory.
See [CMM §3 and Appendix A](https://arxiv.org/html/2610.07907v1).

Its maze attention analysis routes STM reads to the zero-valued sink; authors
suggest FFN expressivity may explain the maze gain. That result should not be
used as evidence of long-term storage on the present task. The linked CMM
repository returned404 on this check, so these CMM claims are paper-based;
no claim is made to have audited its released implementation.

## What v13 changes, and what remains unresolved

    nu_next = nu + g*((rho-1)*nu + beta*kappa*candidate)
    explicit_retention = 1-g*(1-rho)

g=0 keeps activity exactly; g=1 recovers the old fully open update. The same gate
scales decay and writing, with its own learned parameters. This differs from
two independently controllable forget/input gates. Histories and readout still
advance. Gate initialization remains .5, not zero. Existing kappa is retained
to reproduce the tested intervention; no arbitrary-gate constant-write claim
is made. No independent SSM/CMM store has been reintroduced.

For a learned cue direction w, m=w^T*nu, its disturbance is

    m_next-m = w^T [g*((rho-1)*nu + beta*kappa*candidate)].

This is our diagnostic derivation: useful preservation requires a small
disturbance along the cue direction during irrelevant inputs. Small total
activation changes or larger RMS alone cannot establish it. The full derivative
also contains gate and candidate derivatives, beyond diag(1-g*(1-rho)).

The completed matched offline adaptation used one checkpoint (62,400 records,
925 updates), three paired seeds, 500 updates each, B16/T32 complete episodes,
the original WM loss/optimizer and frozen actor/critic. It was not online policy
training. Mean results are reproduced from the saved reviewed result artifact:

| Held-out metric | Adapted old recurrence | Adapted v13 recurrence |
|---|---:|---:|
| t30 activity cue probe |79.29%|81.55%|
| t30 deter cue probe |80.10%|82.85%|
| activity RMS |0.3929|0.4683|
| prior correct/wrong ranking |50%|50%|
| positive prediction >0.1 |0%|0%|
| terminal reward NLL on replay |1.3752|1.3737|

The memory difference is a small trend from one starting checkpoint; paired
per-seed activity intervals include zero. In the new branch, activity probe
accuracy drops from98.22% at t2 to81.55% at t30. Neutral-step gates average0.5304,
close to cue-step0.5322: this does not show learned selective closure. Sampled z
has50% cue probe accuracy at t30 while deter has82.85%; nevertheless, raw activity
already reaches reward directly, so stochastic z cannot alone explain failure.

Evidence: [protocol](C:/Users/Administrator/Desktop/CTMNavigation/tmp/v12_identity_adaptation_control_20261023/protocol.json),
[reviewed results](C:/Users/Administrator/Desktop/CTMNavigation/tmp/v12_identity_adaptation_control_20261023/reviewed_results.json),
and [reward references](C:/Users/Administrator/Desktop/CTMNavigation/tmp/v12_identity_adaptation_control_20261023/reward_reference.json).
These figures describe that completed offline control, not the currently running
v12 process, a newly trained v13 policy, or a live success-rate measurement.

## How reward can become useful

The task rule is r_terminal=cue*(2*a-1), with cue in{-1,+1}, a in{0,1}; other
records have zero reward. The head needs cue, action, and terminal phase jointly.
Current first-layer structure is already capable of an action-specific cue map:

    h0 = SiLU(LayerNorm(U*[projection,z] + W_a*activity))
    h1 = existing_hidden_layer(h0)
    logits = existing_255_bin_output(h1)

If a cue is linearly decodable from activity, W_0 and W_1 can map that direction
with opposite signs. This is representational capacity, not evidence that the
trained bank or downstream layers do so. LayerNorm and other varying features
can still change the effective cue-to-output relationship.

In the new matched control, an external held-out reward-sign probe on single-draw
prior h0/h1 reaches67.15%/63.51%. The actual output nevertheless prefers action0
for every paired history. This places an additional problem between partially
usable hidden features and actual logits, alongside incomplete memory. It does
not prove the last layer alone is responsible: a newly fitted probe and the
jointly trained head solve different optimization problems.

For the inherited distribution, the output value is

    symexp(sum_j softmax(logits)_j * symlog_bin_j).

Roughly balanced positive/negative probability mass produces a value near0.
For balanced terminal labels, knowing phase but ignoring cue gives optimal
two-hot NLL1.366756; knowing the label gives target entropy0.673609, not0.
Terminal records are1/32 of complete episodes. Current terminal NLL near1.37 is
consistent with a weak cue-independent solution. Neither correct signs nor
low all-record loss demonstrates learned action discrimination. The original
replay audit found zero label/action alignment errors.

Recommended next controls, not enabled by this revision:

1. Keep new CTM fixed and train only the existing reward head on the unchanged
   replay distribution and NLL. Compare the same budget to its starting head on
   held-out cue/action groups. If ranking improves, head adaptation/feature
   nonstationarity is implicated; no improvement leaves feature quality and
   optimization unresolved. An optional last-output-layer-only control isolates
   the partial hidden readout, with no architecture change.
2. Evaluate a fixed cue decoder across t2..t30, alongside separately fitted
   decoders. Use identical action/distractor prefixes for opposite cues. This
   distinguishes cue loss from representation drift that refitting can hide.
3. Measure reward gradients on cue-carrying activity/gates and their alignment
   with observation/continuation/KL gradients. Compare zero and terminal records.
   Sparse target frequency is a plausible pressure, not yet a demonstrated
   causal explanation. Avoid simultaneous head/loss/backbone changes.

Preservation, accessibility and supervised use must all work. More persistent
activity alone does not establish a reward model that can train the policy.
