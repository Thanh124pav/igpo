# InGPO pipeline and bug notes

This note documents the runtime pipeline for the repository's InGPO implementation
and records the algorithmic issues fixed in this change.

## Runtime pipeline

### Shared setup

1. Shell entrypoints such as `scripts/train_ingpo_tree_MATH.sh` compose an
   experiment config, merge the requested tree-shape overlay, and call the common
   `ingpo_run` launcher.
2. `configs/ingpo_defaults.libsonnet` defines the InGPO knobs. The default path
   is `algorithm_mode: 'budget_allocation'`, while `share_prune` remains as an
   optional legacy-compatible sibling-local mode.
3. `configs/ingpo_overlay.libsonnet` maps those knobs into
   `InGPOInferenceStrategy`, and the episode generator consumes the annotated
   tree (`ingpo_action`, `ingpo_share_target`, `ingpo_reward_variance`, budget
   metadata) to build training episodes.

### Budget-allocation mode (default)

1. Build the root node for a problem and initialize the frontier.
2. For every depth, collect expandable frontier nodes and compute the requested
   node budget as:

   ```text
   total_depth_budget = base_branch_factor(depth) * number_of_expandable_nodes
   ```

3. Estimate each frontier node's local reward variance:
   - expand first-phase subnodes from the current prefix;
   - expand second-phase continuations from those subnodes;
   - score the conditional matrix `log P(continuation_k | prefix_i)` once per
     unique `(prefix, continuation)` pair;
   - softmax each matrix row to get sampled conditional distributions;
   - compute pairwise total variation (TV) and convert it into a
     simulation-lemma reward variance.
4. Allocate branch factors from the requested depth budget with
   `allocate_branch_factors()`. Each node weight is
   `(sigma_i^4 + lambda) ** 0.25`, where `sigma_i^2` is the estimated reward
   variance. The allocator intentionally uses floor rounding and records any
   underallocated budget.
5. Reuse selected first-phase TV candidates as real child prefixes, expanding
   extra candidates only if the allocation exceeds the reusable candidates.
6. Complete selected candidates, score leaves with the task reward function, and
   carry unfinished children into the next frontier.
7. Attach tree-level statistics, timing, budget, TV support, and reward-variance
   metadata for logging and episode generation.

If `root_allocation` is enabled, step 3-4 runs once across all minibatch roots
before individual trees are built, then each tree receives its precomputed root
branch allocation.

### Share/prune mode

1. Build an SPO-style tree recursively.
2. After generating a sibling set, attach InGPO metadata to each child
   (`ingpo_segment_id`, parent id, depth, default action, and leaf/reward state).
3. Run sibling-local TV SHARE/PRUNE gates before probe expansion:
   - choose cheap-score-nearest sibling pairs within the configured pair budget;
   - score both sibling prefixes on the same sampled continuation support;
   - SHARE a source sibling into a target sibling when the simulation-lemma value
     bound is within `epsilon`;
   - PRUNE low-impact siblings using sibling probability weights and pairwise TV
     bounds.
4. Probe and recursively expand only children whose action is still `expand`.
5. Aggregate rewards and InGPO stats after descendants finish.

## Bugs found and fixed

### 1. SHARE/PRUNE gates were run before child metadata existed

The sibling-local gate filters candidates with `ingpo_action == 'expand'` and
records `ingpo_segment_id` in every decision. The previous DFS flow called the
gate immediately after node expansion, before fresh children received those
fields. As a result, fresh siblings were invisible to the gate and SHARE/PRUNE
usually did nothing.

The DFS now annotates child metadata and terminal leaf state before invoking the
local gate.

### 2. SHARE/PRUNE decisions could be overwritten

After the old early gate location, the next loop unconditionally reset
`child['ingpo_action'] = 'expand'`. Any successful SHARE/PRUNE decision would
therefore be lost before probe expansion and recursion.

The DFS now uses `setdefault()` during metadata initialization and skips probing
for children whose action is no longer `expand`.

### 3. Conditional TV default omitted the total-variation half factor

The conditional-TV estimator softmaxes matrix rows into probability vectors.
Total variation between two distributions is `0.5 * L1(p, q)`. The previous
default left the half factor disabled via `tv_includes_half_factor: false`, which
could produce TV values up to 2 and inflate simulation-lemma variance/budget
weights.

The default is now `true` in both code and config, while the switch remains
available for ablation compatibility.

### 4. Budget-allocation probes could exceed the normal node token budget

In budget-allocation mode, first-phase TV probes are reused as real child
prefixes. When `tv_subnode_max_tokens` was larger than the normal per-node `M`,
those reusable candidates could already be longer than a standard SPO-tree node;
then completion added at least one extra token. The first-phase TV probe budget is
now capped by `M` before probes are generated or extra budget candidates are
sampled.

### 5. Completed budget candidates stayed on the frontier

When a reused first-phase candidate needed a continuation, `_complete_candidate()`
appended the continuation but did not re-check `finish_reason`. A continuation
that stopped naturally could remain `leaf = False`, causing the next
budget-allocation depth to estimate variance and allocate budget to an already
finished answer. Continuation completion is now re-scored as a leaf immediately
when `finish_reason != 'length'`.

### 6. Config/init failures around `ingpo` and `store_logprobs`

The InGPO overlay used to read `$.ingpo.*` directly, so composing
`ingpo_overlay.libsonnet` without first composing `ingpo_defaults.libsonnet` could
fail while the policy-iteration runtime was being initialized. The overlay now
imports default InGPO values itself and exposes them through an additive `ingpo+`
field before mapping them into the inference strategy.

The overlay also sets `node_expander.store_logprobs: true` because budget
allocation ranks and accounts for reusable TV candidates using generation
logprobs/token counts. `NodeExpander` now accepts that config key, and the
efficient IID expander requests/stores generation logprobs when it is enabled.

### 7. Zero-token expansion and missing `chain_of_thought`

When context accounting produced `max_tokens <= 0`, the efficient IID expander
could still attempt generation. Some backends then returned no
`chain_of_thought`, which propagated as missing-node data later in budget
allocation. The expander now skips zero-token expansion before calling the
backend, and budget-allocation candidate completion marks no-token-budget
candidates as terminal instead of sending them to the next frontier.
