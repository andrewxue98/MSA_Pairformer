# MSA Pairformer screening optimization plan

Reviewed 2026-10-26. Upstream: `yoakiyama/MSA_Pairformer`, commit
`875363570df1ae484cf725aba382790444223005`. Fork:
<https://github.com/andrewxue98/MSA_Pairformer>.

This is an implementation plan and code audit, not an optimized model release.
The model source and weights are unchanged. No prediction or GPU performance
benchmark was run for this review. Small CPU probes confirmed specific helper
defects and storage behavior; see [results](audit/probe_results.json).

## Objective and output contract

Make repeated contact screening fast and memory efficient, including monomers
and assemblies with two or four identical chains. Preserve the selected MSA,
query row, sequence order, chain boundaries, masks, weights, query-bias behavior,
contact-head definitions, and requested contact probabilities.

Two distinct claims must never be conflated:

1. **Exact compatibility:** the same finite output values, dtype, shape, ordering,
   and screening decisions on a pinned software/hardware/backend configuration.
   Require exact equality, with a byte-level check if claiming bitwise identity.
2. **Numerical equivalence:** unchanged mathematical computation, but different
   GEMM tiling, accumulation, fusion, or BF16 rounding. Measure differences and
   label them explicitly. These candidates cannot silently become the default
   for a user requiring unchanged outputs.

An optimization enters the exact screening path only after its exact tests pass.
A mathematically equivalent chunked implementation that fails exact equality
stays experimental; do not relax tolerances after seeing the result. Cross-GPU,
cross-version, and cross-backend bitwise equality is not promised.

Keep the existing embedding/logit/training APIs available. Add a contacts-only
inference entry point with an explicit output contract; omitting unrequested
embeddings is not a claim that the complete `forward()` result is identical.

## What the review found

See the [code review](screening-code-review.md) for source locations, evidence,
risks, and additional opportunities. The highest-priority work is:

- Stop once all requested contact layers have run; emit each contact map at its
  producing layer. Avoid extra trunk layers, final MSA update, and language head.
- Chunk MSA and pair transitions, including their normalization, projection,
  activation, and residual write. This removes the largest avoidable buffers.
- Repair differential outer-product chunking and its helper before enabling it.
- Tile initialization and triangle workspaces; avoid whole-pair FP32 copies.
- Release unused representations and query views; stop notebook-wide tensor
  conversion and allocator flushing between screening jobs.
- Reuse a loaded model and cache deterministic preprocessing with complete keys.

### Memory arithmetic, not measured peaks

For batch size 1, BF16 pair width 256 and MSA width 464, with depth 512:

| Total residues | One pair tensor | One MSA tensor | Unchunked pair FFN input projection | Same projection, 64 query rows |
| ---: | ---: | ---: | ---: | ---: |
| 975 | 0.453 GiB | 0.431 GiB | 3.626 GiB | 0.238 GiB |
| 1,950 | 1.813 GiB | 0.863 GiB | 14.505 GiB | 0.476 GiB |
| 3,900 | 7.253 GiB | 1.726 GiB | 58.022 GiB | 0.952 GiB |

The FFN projects to 2,048 channels before SwiGLU. Gate activation and product
buffers coexist with that projection. Chunking does not eliminate the resident
pair state, triangle workspaces, MSA state, output buffers, or allocator overhead.
At depth 1,219, one MSA tensor is 1.027/2.054/4.109 GiB respectively. A full FP32
3,900-by-3,900 contact map is only about 58 MiB per head.

The earlier conversational VRAM ranges were rough estimates for unmodified BF16
inference. They are not a capacity guarantee or a benchmark baseline. Publish
new memory numbers only after measuring the exact implementation and backend.

## Implementation sequence

### Stage 0 — Freeze a trustworthy reference

Deliver `tests/reference/`, fixture manifests, and a benchmark runner before
changing the model. Pin source, checkpoint revision and all checkpoint file
hashes, dependencies, device architecture, precision policy, and backend.
Keep upstream code separately importable in fresh processes. The loader currently
selects the first matching cached files and otherwise follows an unpinned Hub
revision; resolve one coherent snapshot for all three weight files.

Freeze parsed tokens, selected row indices, masks, chain breaks, and random state.
Use synthetic fixtures and licensed public examples in the repository. Keep
user-supplied sequences, downloaded alignments, identities, and prediction files
outside Git. Store only anonymous shape summaries in public benchmark reports.

Use an explicit eval/inference wrapper. Do not apply inference mode to the shared
model `forward()` because pairing/design workflows may require gradients. Keep
the current FP32-parameter/BF16-autocast baseline separate from FP32 arithmetic
validation; changing from an existing FP32 run to BF16 is itself a numerical
change. First measure repeatability of the reference.

**Gate:** reference outputs, state hashes and environment are reproducible;
invalid reference fixtures are identified, not silently repaired. Tests V01–V03.

### Stage 1 — Contacts-only execution and tensor lifetime

Deliver a dedicated screening API, for example `predict_contacts_screening`,
with requested heads, backend, output device and explicit chunk settings.
Default to the caller's already selected MSA; do not impose a new depth limit.

- Execute through zero-based layer 15 for C-beta only, 18 for ConFind or both,
  using the model's configured head indices rather than hard-coded assumptions.
- Apply the existing head immediately after the layer's pair update. Store only
  its small probability map. Preserve sigmoid-then-symmetrize order.
- Skip later trunk work, final MSA update, LM head and representation dictionaries
  only when those outputs are not requested. If all-layer sequence weights or
  embeddings are requested, preserve their requested coverage and continue.
- Avoid D2H/H2D trips for pair embeddings. Release earlier pair references before
  the next layer. Use independent copies for any retained query-only views.
- Make lists/config dictionaries request-local. Do not mutate caller arguments or
  shared query-bias state across jobs. Preserve legacy API behavior separately.

For the released 22-layer stack this removes six later core layers for C-beta
only, or three for both heads, plus the final MSA update. Those layer counts are
not a measured latency reduction. The single-head helpers already stop early;
their remaining savings are lifetime and output handling.

**Gate:** exact contact equality on the same backend, unchanged inputs/state,
correct requested-head behavior, no retained full-MSA view. V04, V05, V13, V15.

### Stage 2 — Chunk independent transitions and contact heads

Deliver inference-only row/token chunking for `PreLayerNorm(Transition)` on both
streams. Preallocate output storage; normalize each tile across the complete
channel dimension and keep all channels in each linear operation. Do not compute
a full normalized input or collect tile results in a list followed by `cat`.
For the MSA, flatten only independent `(batch, row, residue)` positions or tile
without copying noncontiguous tensors. For pairs, tile independent `(i,j)` entries.

Begin with chunk sizes 32, 64, 128; tune from measured memory/throughput. Fuse or
reuse the residual buffer only after proving the old tensor is no longer read
and no requested embedding aliases it. Keep autograd-compatible upstream code
as the training path. Tile head normalization/projection; symmetrize the completed
probability map afterwards or compute both directions for an interface tile.

**Gate:** complete reduction axes preserved; chunked-versus-reference tests at
irregular shapes; strict and numerical results reported separately. V06, V12.

### Stage 3 — Repair and chunk the differential outer product

The active `_chunk()` omits `pair_denom`, and generic `chunk_layer` has additional
confirmed failures. Prefer a small explicit implementation over relying on the
broken general helper; alternatively repair the helper with its full test matrix.

Tile output residue rows, retaining the full MSA-depth contraction and opposite
residue axis. Compute or slice the matching denominator rows, retaining the
original dtype/epsilon/order. Preallocate the pair update, including batch>1 and
tail tiles. Keep global query-biased sequence weighting unchanged. Optional later
tiling of query/key projection must retain the full residue reduction when
computing each row's sequence weight.

Do not cast the denominator to BF16 just to reduce memory or assume every row is
fully covered. Track actual dtypes under CUDA autocast. Stage 3 and Stage 2 can
be reviewed independently, but both depend on Stage 0.

**Gate:** V07 and V08, including masks, supplied weights with query bias off,
all-padding edge cases, and both OPM flavors. Formerly crashing paths get tests
against the working unchunked formula, not against the crash as desired behavior.

### Stage 4 — Initialization and triangle workspace

First build existing positional features in query-row blocks. The current
nearest-bin search creates large int64 distance and one-hot arrays. Direct
integer indices remove this search exactly at the categorical level; replacing
the projection with weight-table addition changes floating-point evaluation and
must be an independently tested candidate. Preserve clipping, offset-100 chain
semantics, feature concatenation order, and the existing bond feature.

For triangles, benchmark the installed cuEquivariance backend and a tiled fallback
separately. Tile normalization/projections and batch independent channels for
contractions while retaining the complete residue contraction dimension. Avoid
materializing whole-pair `.float()` and normalization outputs at once. Audit
contiguous/transposed copies and release intermediates after their last use.
Do not reuse ESMFold2 kernels without verifying this model's weights, gates,
normalization placement and mask semantics.

Cache only small index features or use a bounded cache for immutable initial
features; never accidentally keep several complete pair tensors per shape on
the GPU. Cache keys include all chain-break lists, batch shape, dtype, device,
checkpoint and backend. Working pair tensors must never alias cached storage.

**Gate:** V09–V11, including long-index address arithmetic and memory checks.
No claim of fitting 3,900 residues until a measured full run succeeds.

### Stage 5 — Screening pipeline and repeated-job throughput

Deliver a headless runner with a persistent model, preprocessed-input cache,
bounded CPU queue, explicit job/result ownership, and incremental output writing.

- Cache tokens, selected row indices and masks by input content, parser/filter
  options and versions, seed, crop, query, chain assembly and selected-row hash.
  Cache immutable MSA initial embeddings only for exactly identical inputs and
  weights; later hidden states are query/pair dependent and cannot be reused
  across mutations or different partners.
- Transfer compact tokens or BF16 one-hot input; avoid transient GPU FP32 one-hot
  buffers. Token-index embedding lookup is an optional numerical candidate and
  must preserve every token, including gap/pad/mask and ambiguous amino acids.
  Preserve soft-valued inputs through the general linear projection API.
- Replace Python-level token mapping with a validated lookup table if profiling
  shows CPU preprocessing is material. Preserve selected row IDs, query row,
  tie-breaking and RNG state. Treat greedy-selection accumulator changes as
  numerical candidates because rounding can change a tie and hence the MSA.
- Queue jobs of identical shapes initially. Pad/bucket/batch only after testing
  equivalence to independent runs; a padded row can create a zero denominator
  under BF16, and the masks do not automatically prove padding invariance.
- Export requested maps or exact derived rankings, not the entire hidden-state
  result. Avoid per-item `.item()`, plotting, full-array sorting where a
  tie-preserving selection suffices, repeated `gc.collect()`, `empty_cache()`
  and synchronizations in the hot loop.
- Retry OOM with smaller independent-operation chunks, never a shallower MSA,
  shortened sequence or changed backend without recording that change. A strict
  mode may use only chunk configurations already certified for exact output.

**Gate:** V13–V17. Measure end-to-end throughput including preprocessing, transfer,
serialization, and warm persistent-model runs; quantify cold start separately.

### Stage 6 — Optional compiled/fused specialization

Only after profiling Stages 1–5: test compiled fixed-shape transition blocks,
fused SwiGLU/residual operations, and CUDA graphs for repeated identical shapes.
Prefer selective compilation to compiling CPU I/O, dictionaries, or the entire
dynamic pipeline. Include compiler warmup, graph buffers, cache size and shape
recompilations in the report. Require a measured benefit and the same output
gate; retain eager fallback. Do not assume training checkpointing reduces
inference memory: there is no backward activation tape to discard.

## Explicit exclusions from output-preserving optimization

MSA subsampling/deduplication, lower coverage, cropping, changing query bias or
pairing, omitting residue interactions, computing a monomer once and tiling its
contacts, pruning triangle contraction indices, replacing the head, INT8/FP8,
and stopping before a requested head all change the problem or arithmetic.
Reducing depth to 512 is a separately specified scientific experiment, not an
equivalent implementation of the 1,219-row input.

Keep existing chain encoding quirks and weight-override bugs isolated from
performance commits. Fixing biological semantics may be warranted, but it changes
the reference and requires a new versioned baseline.

## Acceptance and deliverables

Follow the [validation plan](screening-validation-plan.md). Each implementation
commit carries its unit tests, reference comparisons, per-stage allocation/time
measurements and an explicit compatibility label. Provide an ablation table
with one optimization at a time, then combined. Promote only configurations
that meet the relevant unchanged-output contract. A lower memory ceiling can
justify a slower explicit low-memory profile; do not label it a throughput win.

Suggested review units: (1) reference harness, (2) contacts-only path,
(3) transition chunking, (4) OPM and helper repair, (5) initialization,
(6) triangle workspace, (7) persistent runner and preprocessing,
(8) optional fused/compiled kernels. No fixed speedup or peak-VRAM target should
be advertised until those measurements exist.
