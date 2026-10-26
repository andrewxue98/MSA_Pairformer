# Tests and benchmarks for unchanged screening outputs

Companion to the [implementation plan](screening-optimization-plan.md) and
[code review](screening-code-review.md). These are test cases to implement with
the corresponding optimization, not a report of completed model validation.
Only the small [CPU audit probes](audit/probe_results.json) have run so far.

## Reference and acceptance rules

Freeze upstream commit `875363570df1ae484cf725aba382790444223005`, a single complete
checkpoint snapshot and its hashes, Python/PyTorch/CUDA/cuEquivariance versions,
device architecture, autocast and TF32/reduced-precision-reduction policies,
backend selection, thread count, eval mode, and input tensors. Record actual
kernel eligibility/fallback rather than just a package being importable.

Run reference and candidate in separate processes with the same inputs and
weights. First repeat reference inference to establish determinism. If reference
repeats differ, report that fact and investigate deterministic settings; a
candidate cannot be certified bitwise-identical by choosing one convenient run.
Do not change flags between reference and candidate to manufacture agreement.

For the exact profile, require:

- Identical output keys for the API being compared, shapes, dtypes and explicitly
  promised devices. Compare contacts-only output to those keys of upstream,
  rather than pretending omitted embeddings/logits are unchanged.
- Identical finite masks, no new NaN/Inf, and exact values. Also compare raw bytes
  when claiming bitwise identity, since numerical equality alone does not
  distinguish signed zero. A reference that produces nonfinite probabilities
  is an invalid screening fixture, not a successful equivalence result.
- Exactly the same threshold decisions, selected residue pairs, tie behavior,
  intra/inter-chain labels and candidate ranking under the same scoring function.
- No mutation of input tokens, masks, supplied weights, request lists, model
  parameters or cache entries. Preserve relevant RNG consumption/state.

For any failed exact comparison, report maximum/mean absolute error, RMSE,
relative error away from zero, counts of changed values, score margins around
thresholds, changed selected pairs and rank swaps. Diagnose from the first
diverging layer. Correlation alone is never sufficient: a high correlation can
hide a changed screening hit. Numerical-equivalence profiles need a separately
specified tolerance contract before promotion; they do not pass the exact gate.

## Fixtures

Use three fixture families:

1. Deterministic synthetic tensors and MSAs: D in {1, 2, 7, 33}; L in
   {1, 7, 31, 32, 33, 63, 64, 65, 127, 129}; batches 1 and 2; uneven lengths;
   noncontiguous inputs; gaps, ambiguous amino acids, mask and padding tokens;
   varying chain breaks; signed/zero and high-dynamic-range numeric tensors.
2. Licensed public A3M examples with pretrained weights, including monomer,
   heterodimer and repeated-chain examples; freeze exact selected row indices.
   Include diverse/gappy alignments and shallow/deep coverage. Do not regenerate
   MSAs over the network during a regression test.
3. A local, untracked workload: full-length 975/1,950/3,900 residue inputs at
   depths 256/512/1,024/1,219. Preserve the first query row and chain copies.
   The depth series is a benchmark matrix, not a claim that depths have equal
   biological outputs. Each candidate compares only to its matching reference.

Use pairwise coverage of the small shape/flag matrix, plus explicit edge cases;
do not run the full Cartesian product of every size and feature. Large stress
tests are gated/manual GPU jobs, not the default CPU CI suite.

## Test inventory

| ID / proposed file | Cases and assertions | Stage |
| --- | --- | --- |
| V01 `test_reference_and_loading.py` | Pin all three checkpoint files to one snapshot; reject mixed hashes; CPU-load/one-transfer matches current state dict exactly; stale unrelated cached snapshots cannot win; unknown/missing tensors fail; offline replay works. Record reference-repeat determinism and actual backend. | 0, 5 |
| V02 `test_input_contract.py` | A3M lowercase insertions, dots/stops, wrapped FASTA, gap/pad/mask/ambiguous tokens, unsupported characters, inconsistent row widths. Exact parsed tokens and selected IDs versus upstream for supported inputs. Assert full requested length; expose crop/token-cap behavior rather than silently repairing it. Query remains row zero. Frozen hhfilter output is replayable with binary/version recorded. | 0, 5 |
| V03 `test_reference_contacts.py` | Pretrained full forward versus existing C-beta, ConFind and combined helpers, FP32 and BF16 treated as separate baselines. Save both directed head probabilities where diagnostic hooks permit and final symmetric maps. Whole reference artifacts are unmodified and independently generated. | 0 |
| V04 `test_contact_early_exit.py` | C-beta only, ConFind only, both, custom/equal/reversed configured head layers; exact requested maps. Hooks assert layer counts (16/19 for released indices), no LM/final-MSA execution when unrequested. Requests for final representations or all-layer weights must continue or explicitly use the legacy API. | 1 |
| V05 `test_output_lifetimes.py` | Compare immediate-head maps to deferred-head maps. Output dictionaries contain no unrequested pairs/MSAs. Check storage bytes and aliases for query-only outputs, including `store_*_cpu` combinations. Deliberate weak references and post-return allocation measurement confirm intermediates can be released. Check results survive the next inference unchanged. | 1 |
| V06 `test_transition_chunks.py` | Pair width 256 and MSA width 464 with real pretrained weights; chunks 1/31/32/64/128/larger-than-input; uneven tails, batch 2, strides, eval dropout. Compare full PreLayerNorm+FFN+residual, not just a linear layer. Detect untouched output elements using sentinels. Run exact and diagnostic comparisons. Training path and RNG remain unchanged. | 2 |
| V07 `test_outer_product_chunks.py` | Differential and vanilla OPM, batch 1/2, nonuniform masks, gaps distinct from padding, query bias on/off, provided weights with `seq_weights_dict=None`, masked rows/columns and zero pair coverage. Compare weights, denominator tiles and final pair update to unchunked computation. Keep full depth reduction. Missing `pair_denom` regression must be exercised. | 3 |
| V08 `test_chunk_helper.py` | Single/multiple inputs, one/two batch axes, broadcast and noncontiguous inputs, low_mem mode, tuple/dict outputs, preallocated `_out`, additive mode, singleton/tail chunks, invalid/zero chunk sizes. Verify every output element and layout contract. Cover the four failures in the audit probes. | 3 |
| V09 `test_initialization.py` | No break, one break, three breaks, boundaries at first/last valid index and differing breaks across batches; clipped offsets around -33/-32/32/33 and multiple +100 shifts. Exact category indices and separate comparison of projected features. Preserve current cross-boundary bond values. Cache hits/misses and mutation cannot corrupt initial state. | 4 |
| V10 `test_pair_and_triangle_kernels.py` | Incoming/outgoing triangles separately, masked/unmasked and noncontiguous tensors, projection/norm/gating/residual checkpoints, real weights, both supported backends. Pair-weighted averaging with arbitrary full masks; every query chunk still sees all keys. Compare same-backend first, then explicitly label cross-backend differences. | 4 |
| V11 `test_long_shapes.py` | Target pointer/index boundary calculations for L=2,896 and 2,897 at pair width 256, then L=3,900. Audit each larger expanded tensor's address bounds separately. Validate boundary rows against independently sliced/reference contractions retaining full reduction axes. Perform full long-input validation when reference hardware permits. No assumed safety from small tests. | 4 |
| V12 `test_contact_head_chunks.py` | Exact probabilities and sigmoid-before-symmetrization; asymmetric directed logits, extreme logits, chunk tails, heads with real weights. Interface-only export equals the matching block of the full map, using both orientations. Do not infer one triangle from a nonsymmetric representation. | 2, 4 |
| V13 `test_screening_memory.py` | Warm and repeated same-shape jobs, alternating shapes, 100 small jobs, output retention intentionally on/off, CPU-offload paths and cache eviction. Check allocated memory returns to stable baseline after outputs release; distinguish allocator reserve from leaks. Capture maximum per-stage allocations and raw storage ownership. | 1, 5 |
| V14 `test_preprocessing_cache.py` | Exact token lookup versus parser vocabulary; explicit rejection/fallback for soft inputs; same MSA with different query, filtering, seed, chain multiplicity, breaks or parser version must miss cache. Four identical chains remain four with three boundaries. Same selected homolog rows and order across reference/candidate assemblies. Test batching/padding against per-job outputs, not just shape validity. | 5 |
| V15 `test_api_and_state.py` | Existing public helpers and general forward, fresh/reused models, caller-list immutability, mutable constructor defaults, sequence-weight precedence, eval/training and grad-enabled fallback, concurrent-request serialization, CPU import with/without optional CUDA packages. Global NumPy/Torch RNG state and cache contents recorded. Behavioral bug repairs versioned separately. | 0–5 |
| V16 `test_screening_decisions.py` | Fixed thresholds (including exact ties), top-L/top-L/2 and top-k with pinned tie policy, short sequences/no eligible pairs, all equal scores, NaN policy, chain-local separation and inter-chain pairs. Exact candidate ordering under a predefined score. Partition selection must preserve the existing threshold-and-all-ties behavior. | 5 |
| V17 `test_runner_recovery.py` | Job ordering, deterministic serialization/round-trip, restart/resume, cache invalidation, bounded prefetch, OOM cleanup, fallback to certified chunks only, no automatic MSA trimming/backend changes. Failed job is recorded without contaminating later jobs. Fixed-shape compiled/graph path versus eager is gated identically. | 5, 6 |

## Additional test design details

### Preserve complete mathematics while tiling

- FFNs and contact heads may tile independent positions, but each LayerNorm
  keeps its entire channel dimension.
- OPM tiles output residues, not the sequence-depth reduction. Pair denominators
  must correspond to the tile and use the same weighted masks and epsilon.
- Pair-weighted averaging may tile MSA rows or query positions, but softmax and
  aggregation retain all residue keys. Empty/padded domains need explicit tests.
- Triangle contraction always retains every residue k. A channel/output-row tile
  is not permission to truncate interactions or use banded/sparse attention.
- Compute both contact orientations before symmetrizing probabilities. Do not
  optimize away the final head's nonlinear order.

Build tests around independent unmodified reference functions and frozen outputs,
not around a second copy of the optimized algorithm. Random synthetic module
tests are necessary but insufficient: real trained weights can expose saturation,
ranking sensitivity and cancellation that small random weights miss.

### Bitwise and screening-level checks

For exact profiles, use zero tolerance and raw-byte comparison as appropriate.
Record aggregate diagnostics even on an exact failure. Candidate ranking and
top-contact agreement are additional tests, not substitutes for probability
equality. All decisions must use the same score definition, float export dtype,
threshold comparison (`>` versus `>=`), tie handling and chain indexing.

If a numerical candidate passes many examples but fails a near-threshold case,
report it; do not claim output preservation from an average metric. A rerun of
borderline cases through the reference can be an explicitly separate screening
policy, but it does not make approximate maps identical or guarantee unchanged
screening without a justified error bound.

### Large reference runs that do not fit

Use original unchunked trained modules on independent output-row slices while
keeping full reduction axes for targeted checks, then obtain a complete upstream
long-input reference on a sufficiently large device if exact full-tetramer
certification is required. A patched low-memory reference is not independent
evidence for the very patch under test. If full reference execution is unavailable,
report module/small-end-to-end evidence and the remaining certification gap;
never label the tetramer exact solely because small examples pass.

## Benchmark protocol

Measure four distinct phases: cold model load, preprocessing/cache behavior,
GPU inference, and export/serialization. Report total wall time as well as
CUDA-event time. Exclude network downloads from model compute benchmarks and
record them separately for cold-start workflows.

1. Check GPU capacity and other active jobs without terminating unrelated work.
   Record hardware, driver, clocks/power mode when available and initial free VRAM.
   Do not compare a contended run to an idle run as an optimization effect.
2. Run each configuration in a fresh process to compare peaks. Warm up at least
   twice, separate compilation/autotuning time, then time at least five repetitions
   for small/medium shapes; use a documented smaller count for very long runs.
   Synchronize at measurement boundaries, not inside the production hot loop.
3. Record peak allocated and reserved CUDA memory, current post-run allocation,
   system/process GPU usage if available, peak host RSS, and D2H/H2D bytes/time.
   CPU offload can lower VRAM while raising host RAM and latency. Report both.
4. Profile one representative run with shapes and memory attribution. Use separate
   unprofiled repetitions for timing to avoid profiler overhead in throughput.
5. Compare baseline, each isolated change, and the cumulative implementation.
   Sweep independent-operation chunk sizes without changing data or precision.
   Include the generic full forward, existing single-head helpers, and new
   contacts-only path so the baseline is appropriate to the requested outputs.
6. Benchmark the local full-length shape matrix at D=512, then depth sensitivity
   256/1,024/1,219 as resources permit. Also benchmark many shorter candidates,
   because a low-memory setting can hurt throughput on small proteins.
7. Test persistent-worker throughput, including same/alternating shape sequences,
   cache cold/warm cases, output writing and controlled batch sizes. Include the
   cache's VRAM/RAM and graph/compile storage in peak accounting.

The machine previously reported about 95.6 GiB total GPU capacity; available
memory was much smaller during the earlier discussion. Re-query at execution
time. These are contextual hardware observations, not benchmark results.

Suggested report fields:

```text
source_sha, checkpoint_hashes, environment_id, backend, precision_policy
input_hash, selected_row_hash, B, D, L, chain_breaks, requested_heads
chunk_settings, cache_policy, compile_or_graph_mode, repetition_count
load_s, preprocess_s, h2d_s, inference_s, export_s, total_s
peak_allocated_GiB, peak_reserved_GiB, host_peak_RSS_GiB, post_release_GiB
exact_values_equal, raw_bytes_equal, nonfinite_counts, max_abs_error, rmse
threshold_flip_count, selected_pair_changes, candidate_rank_changes
status, failure_stage, error_type
```

No model weights, biological identity, raw user MSA or query sequence belongs in
public benchmark reports. Input hashes and chain metadata can remain in local
manifests; public fixtures use synthetic or explicitly public inputs.

## Rollout gates

Land reference tests first. Land output-lifetime/early-exit changes next, then
independent chunked modules one at a time. Every module change needs relevant
unit tests plus pretrained end-to-end comparisons before integration. Follow with
full workload memory/throughput measurements and a repeat-job leak test.

Default CPU CI uses tiny fixtures and no network/checkpoints. Optional GPU CI
uses pinned local checkpoints and a declared backend. Long-input/large-memory
tests are separate jobs. Mark unavailable hardware as skipped with the reason,
not passed. Existing gradient paths must retain their tests when inference-only
buffers are added. Export the final compatibility matrix: supported shapes,
dtypes, backends, exact/numerical status, memory peaks, and known limitations.
