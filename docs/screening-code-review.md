# Screening runtime and memory code review

Scope: upstream `875363570df1ae484cf725aba382790444223005`, reviewed 2026-10-26.
Source links below are pinned to that revision. Reviewed the inference model,
core/outer-product/triangle layers, positional encoding, contact heads, chunk
helper, MSA parser/selection/masks, package dependencies, contact ranking,
notebook inference/conversion/cleanup, and selected pairing/variant workflows.
Training and unrelated structure-download utilities were surveyed, not exhaustively
audited. No trained-model prediction or GPU benchmark was performed.

Priority means implementation priority for screening. **Confirmed** means a CPU
probe reproduced behavior; **static** means direct source evidence;
**candidate** means a benefit/equivalence still needs measurement.
The [plan](screening-optimization-plan.md) defines exact versus numerical gates.

## Main inference path

### R01 — P1: Run only the layers needed by requested contact heads

**Static.** [`model.py:535–573`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/model.py#L535)
runs the entire core for both heads, although the released heads use zero-based
layers 15 and 18. General `forward()` also evaluates the LM head. Stop at the last
requested layer and emit contacts immediately in a dedicated inference API.
Preserve the original path for final embeddings/logits and all-layer weights.
The existing single-head helpers already implement early stopping; do not claim
the same saving twice. Tests V04/V05.

### R02 — P1: Avoid retaining or round-tripping multi-GiB embeddings

**Static.** [`model.py:215–255`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/model.py#L215)
stores requested pair snapshots and returns a final pair even when contact-only
callers discard it. `forward()` forces pair snapshots onto GPU when contacts are
requested (`:490`); `predict_contacts()` instead copies snapshots to CPU and then
back for the heads (`:566`). At 3,900 residues, each BF16 pair is 7.253 GiB.
Stream head results and keep only requested maps. Avoid a list/stack of pair
chunks that recreates the peak at concatenation. Tests V05/V13.

### R03 — P1: A query-only view can retain the whole MSA

**Confirmed storage behavior; static call-site bug.** At
[`model.py:251–254`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/model.py#L251),
the `store_msa_repr_cpu=True, query_only=True` branch returns `msa[:, :1]`
without `.cpu()`. This view still owns the storage of the full MSA. Other saved
query views can also retain backing storage when kept on GPU. Clone or move
only the needed view, or return nothing in the contact-only API. Honor existing
dtype/device contracts explicitly when fixing the misleading CPU flag.
The probe demonstrates storage retention without loading the model. V05/V13.

### R04 — P1: Unchunked transitions create the largest removable buffers

**Static dimensions; candidate implementation.**
[`core.py:33–74`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/core.py#L33)
projects each vector to eight times its width. SwiGLU then allocates activation
and product intermediates. At 3,900 residues the pair projection is 58.022 GiB
in BF16. Tile independent MSA positions and pair entries, including normalization.
Preallocate outputs; add residuals only after the old tile is no longer read.
GEMM shape changes can alter rounding, so mathematical independence alone does
not establish bitwise equality. Tests V06/V12.

### R05 — P1: Differential OPM chunking is currently unusable

**Confirmed.**
[`outer_product.py:197–229`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/outer_product.py#L197)
defines `_opm(a,b,pair_denom)` but `_chunk` supplies only `a,b`; enabling it raises
`TypeError`. Forward also omits the denominator in the `_chunk` call (`:297`).
Pass the corresponding denominator tile and retain the entire depth reduction.
Repair batch/tail behavior together; avoid changing epsilon or mask handling.
The CPU probe invokes the real extracted `_chunk` with a signature-compatible
stub; it is not a full OPM numerical test. V07/V08.

### R06 — P1: Generic chunk helper has more than one correctness defect

**Confirmed.**
[`chunk_layer.py:99–125`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/chunk_layer.py#L99)
computes flattened batch size by multiplying first dimensions across input keys,
rather than the dimensions of the broadcast batch shape. Two inputs or multiple
batch axes produce incorrect allocation/reshape behavior. `low_mem=True` calls
undefined `_chunk_slice`. A single-input/one-axis control works. Use explicit
inference tilers or comprehensively repair the utility before wider adoption.
Do not merely thread `pair_denom` through this helper without testing multiple
inputs. V08.

### R07 — P1: Positional initialization materializes unnecessary integer arrays

**Static.**
[`positional_encoding.py:53–91`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/positional_encoding.py#L53)
finds the nearest bin by broadcasting each pair against all bins, then creates
int64 one-hot tensors. Autocast does not reduce integer storage. Categorical
indices are already known after clipping; avoid the nearest-bin search and tile
feature construction. A table-lookup/summed-weight implementation is a separate
numerical candidate because it changes projection evaluation. Preserve the exact
constant residue/chain features and the offset-100 chain behavior. V09.

### R08 — P1: Whole-pair FP32 normalization and triangle copies

**Static.**
[`pairwise_operations.py:127–163`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/pairwise_operations.py#L127)
explicitly normalizes `pair_rep.float()` and `out_rep.float()`, then casts back.
The gated dual projection, mask multiplication, contiguous copies and einsum
output also coexist. BF16 autocast alone does not remove these allocations.
Preserve FP32 reduction accuracy using tiled/fused normalization; profile the
existing cuEquivariance backend before writing another kernel. Contract the full
residue axis even when tiling independent output rows/channels. Different
backends are not automatically output-identical. V10/V11.

### R09 — P2: Pair-weighted averaging can be tiled without removing keys

**Static/candidate.**
[`pairwise_operations.py:335–406`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/pairwise_operations.py#L335)
holds projected values/gates, masked copies, pair-derived attention, and outputs.
Compute pair attention once per block; tile MSA rows for values/gates/output or
tile attention query rows while preserving all keys and the full softmax.
Compare layout-specific einsum/matmul copies before changing layouts globally.
Do not substitute dot-product attention or introduce a new scale. V10.

### R10 — P2: Zero-dropout still performs a full tensor multiplication

**Static.** [`core.py:88–110`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/core.py#L88)
builds structured ones and multiplies the whole input even in eval or when p=0.
An eval/p=0 fast return can remove allocation and bandwidth. Test finite values,
signed zeros, alias behavior, and nonzero-dropout training separately. Do not
change training RNG consumption under a performance patch. V06/V15.

### R11 — P2: One-hot input projection and bond projection are candidates

**Static/candidate.**
[`model.py:428–434`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/model.py#L428)
projects categorical one-hot MSA tokens through a bias-free linear layer and
projects 0/1 adjacency to a full pair tensor. A token lookup can eliminate dense
one-hot buffers; tiled bond addition can avoid a second full pair allocation.
Lookup results must use the same weight casting as autocast and be validated
against GEMM. Soft/probabilistic MSA inputs in pairing/design still need linear
projection. Current adjacency includes consecutive positions across chain
boundaries; changing that is a semantic fix, not a compatible optimization. V09/V14.

### R12 — P2: Process the head before exporting and preserve its order

**Static.**
[`regression.py:54–64`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/regression.py#L54)
normalizes 256-channel pairs and projects to one probability, then symmetrizes.
Tile projection and retain only scalar probabilities. For interface-only output,
compute both directed blocks before averaging. Symmetrizing logits/representations
before sigmoid is not equivalent. Keep the full pair trunk even when exporting
only interface scores. V12.

## Preprocessing, orchestration, and screening throughput

### R13 — P1: Notebook export and cleanup inflate memory and latency

**Static.** In [`MSA_Pairformer_with_MMseqs2.ipynb`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer_with_MMseqs2.ipynb),
`run_msa_analysis` runs general forward, asks for every layer's sequence weights,
and passes the entire result to `convert_to_numpy`. That recursively converts
BF16 tensors to FP32 on GPU before copying to CPU, including large hidden states
that are then discarded. Export only requested maps, moving to CPU before any
storage-format conversion where exact conversion equivalence is verified.
`clear_gpu_memory` scans every GC object, deletes a local loop variable (not other
owners' references), collects three times and flushes/synchronizes CUDA. Replace
this with bounded ownership and reuse, not a per-job allocator reset. V13/V17.

### R14 — P1: Preserve homomer multiplicity in the screening frontend

**Confirmed.** Notebook `prepare_sequences(remove_duplicates=True)` turns four
identical chains into one. A CPU synthetic `ACD:ACD:ACD:ACD` probe confirms it.
The screening frontend must explicitly retain duplicates and construct paired
homomer rows with one column block per chain and breaks at cumulative lengths.
Do not confuse chain deduplication with MSA-row selection. Keep the construction
identical across reference and optimized runs. This is frontend configuration,
not an assertion that the notebook default is an appropriate multimer reference.
V02/V14.

### R15 — P2: Cache parsing/filtering/tokenization without changing selection

**Static/candidate.**
[`dataset.py:150–175`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/dataset.py#L150)
repeats parsing, array conversion, filtering and tokenization per MSA object.
`tokenize_msa` uses Python-level `np.vectorize` (`:237`); greedy selection (`:277`)
reallocates a growing distance matrix, deletes selected columns, and recomputes
means each step. Cache selected indices/tokens with the complete policy key.
An ASCII lookup table can preserve tokenization exactly; incremental distance
sums may alter floating ties and require stronger tests. hhfilter also consumes
global NumPy RNG for a temporary filename (`:313`): preserve/document RNG state
when caching or changing temporary-file handling. V02/V14/V15.

### R16 — P2: Persistent workers and same-shape microbatches

**Candidate, not a benchmark finding.** Reuse one eval model per GPU, bounded input
prefetch, immutable request-local settings and incremental result serialization.
Pin checkpoints, cache compiled shapes and bound every cache. Start with exact
shape grouping; padding and batching can change backend algorithms or masked
normalization, so validate before using them as default. OOM fallback should
change workspace chunking, not biological input. Concurrent query-bias toggles
on a shared model are unsafe; use explicit request configuration or serialize.
V13/V15/V17.

### R17 — P2: Contact ranking can avoid a full sort, but ties matter

**Static/candidate.**
[`utils.py:477–491`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/utils.py#L477)
sorts all eligible scores to select the top-L threshold, then includes every tie.
Partition can find the same cutoff, followed by the same ordered mask extraction.
Do not replace this with an arbitrary exactly-L top-k, which changes tie behavior.
Preserve NaN/error policy and original pair enumeration. V16.

### R18 — P2: Loading and dependency overhead/reproducibility

**Static.**
[`model.py:382–411`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/model.py#L382)
loads checkpoint tensors to the target device, constructs a CPU model, copies
weights into it, then moves the model back to the target. Load one pinned snapshot
on CPU and transfer once; release checkpoint references. The first cached filename
match is not guaranteed to select a coherent revision. Restrict Hub downloads to
needed files and retain hashes. [`pyproject.toml`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/pyproject.toml)
does not declare `huggingface_hub` despite the unconditional model import, while
large optional accelerated packages are mandatory on Linux. Split inference,
acceleration and research extras with explicit backend validation. V01/V15.

## Correctness barriers to keep separate from performance changes

### R19 — CPU imports and supplied-weight behavior

**Static.** `pairwise_operations.py:22–34` only sets `CUEQUIVARIANCE_AVAILABLE`
inside the CUDA branch or an ImportError handler. On a normal CPU-only import it
can remain undefined before class defaults reference it. Initialize it to false
explicitly when enabling CPU tests, recording this compatibility repair.

At `model.py:194–198`, non-None `seq_weights_dict` takes precedence even when
empty; public wrappers default to `{}`, which can suppress supplied `seq_weights`.
Query-biased OPM recomputes weights regardless of the supplied vector. Tests must
record these combinations and not assume a supplied weight was honored. A fix
changes outputs for affected calls; version the behavioral repair separately.

### R20 — Input limits, padding and chain features are not safe shortcuts

**Static.** `dataset.py:155–167` calculates `subset_depth` from `max_tokens`, but
passes `max_seqs` to the selection routine instead. Selection `none` bypasses
depth reduction. The default `max_length=1024` can randomly crop longer assemblies.
Do not trust these settings as a memory admission controller; validate the final
actual tensor dimensions and set full length explicitly for screening.

Masked rows are divided by a coverage count plus `1e-32`
(`outer_product.py:269`), and pair denominators also use that epsilon (`:200`).
BF16/autocast dtype behavior and all-padding rows/pairs must be tested for NaNs;
later masking cannot reliably repair NaNs. Do not change epsilon or masking under
the name of efficiency. Chain breaks are encoded as offsets, not general assembly
symmetry. Avoid symmetry/block-reuse assumptions. V02/V07/V14.

### R21 — Caller mutation and notebook API drift

**Static.** `model.py:481–485` appends contact layers to the caller's list. Core
constructor defaults contain mutable dictionaries and remove `lambda_init` from
one (`:65–73`). Copy configuration/request collections before mutation in a
separately tested repair; check repeated instances and requests. The notebook's
conservation/Jacobian helper passes obsolete `return_contacts=False` to a model
that now accepts `return_cb_contacts`/`return_confind_contacts`. Do not build a
production runner by copying that helper without API contract tests. V15.

## Optional adjacent workloads

### R22 — Gradient-based pairing is a separate execution mode

**Static/candidate.**
[`pairing_optimization/mp_em.py:153–189`](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/pairing_optimization/mp_em.py#L153)
runs a full model to use an intermediate Potts layer; it could request just that
layer. It builds full amino-acid pair couplings before zeroing within-chain blocks.
Computing only needed directed blocks could save memory, but the separate MRF
head is not the contact head and needs its own equivalence suite.
Its per-species Cartesian-product input expansion (`:236–247`) can be tiled while
preserving candidate order and assignment scores. `mp_pdp.py` uses gradients;
never apply inference-only in-place buffers, detached caches or global
`inference_mode` to that path. These are later work, outside the initial contact
screening implementation.

### R23 — Mutation/Jacobian screening should reuse only invariant work

**Static/candidate.** Notebook Jacobian loops clone a full one-hot MSA for every
query mutation and allocate a dense NumPy float64 Jacobian. Use bounded mutation
microbatches and disk-backed output if that workload is requested. Preserve the
same mutations, row weights and output dtype/order; stream reductions only when
the caller asks for a derived result. The Figure 3 mutation script repeatedly
deepcopies tokens, extracts scalar GPU values and rewrites accumulated results.
Cache invariant preprocessing and batch transfers/writes. Do not cache evolved
MSA/pair states across mutated queries: bidirectional updates make them dependent
on the query. Not required for the first contact-screening API.

## Evidence and review limitations

Run `python docs/audit/probe_upstream.py` in an environment with PyTorch.
[Recorded CPU results](audit/probe_results.json) confirm four chunk-path failures,
query-view storage retention and homomer deduplication, plus a passing chunk
control. Source hashes identify the exact files tested. The script extracts the
real OPM chunk method to isolate its call-signature failure; it does not substitute
for trained-module numerical tests. The remaining findings are source-based.
GPU kernel eligibility, numerical drift, output stability and actual speed/memory
benefits remain to be tested as specified in the validation plan.
