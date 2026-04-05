"""
token_fusion.py — BccT Project (CS 534, Team 6)
Author: Corrin

Implements the Token Fusion module from Zhu et al. (2026), "BccT: An Efficient
Transformer Model for Blood Cell Classification."

The module is inserted as a forward-hook **after the FFN** (feed-forward
sub-layer) inside each of the 12 ViT-Base blocks.  It is NOT applied after
the MHSA sub-layer.

Core algorithm (paper Section 3.2):
  1. Separate the [CLS] token (index 0) from patch tokens.
  2. Partition patch tokens into alternating sets A (even) and B (odd).
     — Eq. 3: Sₐ = {X₁, X₃, …}, S_b = {X₂, X₄, …}
  3. Compute head-averaged keys K̄ᵢ = (1/H) Σₕ Kᵢ⁽ʰ⁾  [Eq. 4a]
  4. For every token in A find its most similar token in B via cosine sim
     on K̄.  Collect the r=16 pairs with highest similarity.  [Eq. 4b]
  5. Merge each selected pair using size-weighted average:
         X_merged = (zᵢXᵢ + zⱼXⱼ) / (zᵢ + zⱼ)
     Same formula applied to K̄ and V̄.   [Eq. 11, Eq. 12]
  6. Update size counters: z_merged = zᵢ + zⱼ.
  7. In subsequent blocks, add log-z attention bias column-wise so that
     merged tokens receive proportionally more attention weight:
         logits += log(z)   (column-wise)   [Eq. 5]

The hook carries forward two state tensors per sample across layers:
  • token_sizes  : (B, N_current)    int/float – how many original patches
                   each surviving token represents (starts all-ones).
  • merged_keys  : not stored persistently; re-derived each block from the
                   current block's MHSA key projections.

Integration with HuggingFace ViT
---------------------------------
`patch_token_fusion_hooks(vit_model, r=16)` modifies each ViT block via:
  (a) A *monkey-patch* on each block's ViTSelfAttention.forward that:
      - Captures head-averaged K̄ before computing attention (Eq. 4a).
      - Injects the log-z column bias into QK^T before softmax (Eq. 5).
  (b) A *post-forward hook* on each ViTLayer (whole block = after FFN) that
      performs bipartite matching (Eq. 4b) and token merging (Eq. 11/12).

Both share a `BlockState` object (one per block).

Why monkey-patch instead of hooks for log-z?
  HuggingFace ViT does not expose intermediate attention scores through the
  standard forward-hook API.  The log-z bias must be inserted *between*
  the QK^T matmul and the softmax, which requires wrapping the forward method
  of ViTSelfAttention directly.  This is the same approach used by the
  official ToMe (Token Merging) implementation (Bolya et al., ICLR 2023).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import ViTModel
from transformers.models.vit.modeling_vit import ViTLayer, ViTAttention

# Shared state object — one instance per ViT block

@dataclass
class BlockState:
    """
    Carries mutable state that must be shared between the two hooks that wrap
    each ViT block:
      • `keys`        : head-averaged key tensor captured from MHSA, shape
                        (B, N, head_dim).  Set by _mhsa_pre_hook, read by
                        _layer_post_hook.
      • `token_sizes` : (B, N) int — number of original patches each current
                        token represents.  Maintained across all 12 layers.
    """
    r: int = 16
    keys: Optional[Tensor] = None          # filled by MHSA hook each forward
    token_sizes: Optional[Tensor] = None   # initialised on first call

# Cosine-similarity bipartite matching  (Eq. 4a / 4b)

def bipartite_soft_matching(
    keys_a: Tensor,          # (B, |A|, d_head)
    keys_b: Tensor,          # (B, |B|, d_head)
    r: int,
) -> Tuple[Tensor, Tensor]:
    """
    Eq. 4b — For each token in A find its nearest neighbour in B (cosine sim),
    then select the top-r pairs by similarity score.

    Args:
        keys_a: Head-averaged keys for set A.   shape (B, nA, d_head)
        keys_b: Head-averaged keys for set B.   shape (B, nB, d_head)
        r:      Number of pairs to merge per block.

    Returns:
        idx_a : (B, r) — indices into set A of the selected pairs
        idx_b : (B, r) — indices into set B (best B-match for each A token)
    """
    # L2-normalise for cosine similarity
    a_norm = F.normalize(keys_a, p=2, dim=-1)   # (B, nA, d)
    b_norm = F.normalize(keys_b, p=2, dim=-1)   # (B, nB, d)

    # (B, nA, nB) — cosine similarity matrix
    sim = torch.bmm(a_norm, b_norm.transpose(1, 2))

    # For each A-token, find the best B-match
    best_sim, best_b = sim.max(dim=2)           # (B, nA), (B, nA)

    # Rank A-tokens by their best similarity and pick top r
    # clamp r to the number of available A-tokens
    r_eff = min(r, keys_a.shape[1])
    _, top_a = best_sim.topk(r_eff, dim=1, largest=True, sorted=False)  # (B, r)

    # Gather the corresponding B indices
    top_b = best_b.gather(1, top_a)             # (B, r)

    return top_a, top_b                          # Eq. 4b

# Merge step  (Eq. 11 / 12)

def merge_tokens(
    x:           Tensor,    # (B, N, C) — full sequence including CLS
    keys:        Tensor,    # (B, N_patch, d_head) — head-averaged keys (no CLS)
    sizes:       Tensor,    # (B, N)    — token sizes (includes CLS at [:, 0])
    idx_a:       Tensor,    # (B, r)    — A-indices in patch-only indexing
    idx_b:       Tensor,    # (B, r)    — B-indices in patch-only indexing
    a_mask:      Tensor,    # (B, N_patch) bool — True for set A tokens
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Eq. 11 — Merge selected (A, B) pairs using size-weighted average.
    Eq. 12 — Same operation applied to key tensor.

    CLS token (index 0) is *never* touched.

    Returns:
        x_new     : merged sequence, shape (B, N - r, C)
        keys_new  : merged keys,     shape (B, N_patch - r, d_head)
        sizes_new : updated sizes,   shape (B, N - r)
    """
    B, N, C = x.shape
    device   = x.device

    # Separate CLS from patch tokens
    cls_token = x[:, :1, :]              # (B, 1, C)
    cls_size  = sizes[:, :1]             # (B, 1)
    patches   = x[:, 1:, :]             # (B, N_patch, C)
    p_sizes   = sizes[:, 1:]             # (B, N_patch)

    N_patch = patches.shape[1]

    # Convert A/B indices to global patch-sequence indices
    # idx_a, idx_b are in *patch-only* space (0 … N_patch-1)

    # Convert A-local indices to global patch indices:
    #   a_global[b, k] = the k-th A-token's position in patches[b]
    a_positions = torch.where(a_mask)[1].view(B, -1)  # (B, nA)
    b_positions = torch.where(~a_mask)[1].view(B, -1) # (B, nB)

    # Gather actual global patch indices for selected pairs
    src_idx = a_positions.gather(1, idx_a)    # (B, r) — will be merged INTO dst
    dst_idx = b_positions.gather(1, idx_b)    # (B, r) — destination tokens

    # Size-weighted merge (Eq. 11)
    src_sz = p_sizes.gather(1, src_idx).unsqueeze(-1).float()  # (B, r, 1)
    dst_sz = p_sizes.gather(1, dst_idx).unsqueeze(-1).float()  # (B, r, 1)

    src_x  = patches.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, C))  # (B,r,C)
    dst_x  = patches.gather(1, dst_idx.unsqueeze(-1).expand(-1, -1, C))  # (B,r,C)

    merged_x  = (src_sz * src_x + dst_sz * dst_x) / (src_sz + dst_sz)
    merged_sz = (src_sz + dst_sz).squeeze(-1)                            # (B, r)

    # Write merged result back into dst positions
    patches_new = patches.clone()
    p_sizes_new = p_sizes.clone().float()

    patches_new.scatter_(
        1,
        dst_idx.unsqueeze(-1).expand(-1, -1, C),
        merged_x,
    )
    p_sizes_new.scatter_(1, dst_idx, merged_sz)

    # Eq. 12: same merge for keys
    d = keys.shape[-1]
    src_k = keys.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, d))
    dst_k = keys.gather(1, dst_idx.unsqueeze(-1).expand(-1, -1, d))
    merged_k = (src_sz * src_k + dst_sz * dst_k) / (src_sz + dst_sz)

    keys_new = keys.clone()
    keys_new.scatter_(1, dst_idx.unsqueeze(-1).expand(-1, -1, d), merged_k)

    # Build boolean mask to remove src tokens
    # We want to drop the src positions (they were merged into dst)
    keep_mask = torch.ones(B, N_patch, dtype=torch.bool, device=device)
    # scatter False at src positions
    keep_mask.scatter_(1, src_idx, False)

    # Gather surviving patch tokens
    # We need to re-order; just use the boolean mask row-by-row.
    # All rows have the same number of survivors (N_patch - r) because
    # we always merge exactly r pairs (enforced by r_eff ≤ nA in bipartite_soft_matching).
    patches_out = patches_new[keep_mask].view(B, N_patch - idx_a.shape[1], C)
    p_sizes_out = p_sizes_new[keep_mask].view(B, N_patch - idx_a.shape[1])
    keys_out    = keys_new[keep_mask.unsqueeze(-1).expand_as(keys_new)].view(
        B, N_patch - idx_a.shape[1], d
    )

    # Re-attach CLS
    x_new     = torch.cat([cls_token, patches_out], dim=1)
    sizes_new = torch.cat([cls_size,  p_sizes_out.to(sizes.dtype)], dim=1)

    return x_new, keys_out, sizes_new

# Log-z attention bias  (Eq. 5)

def apply_log_z_bias(
    attention_scores: Tensor,    # (B, H, N_q, N_k) — raw QK^T / sqrt(d)
    token_sizes:      Tensor,    # (B, N_k) — size of key tokens
) -> Tensor:
    """
    Eq. 5 — Proportional attention compensation.

    After merging, some tokens represent more original patches than others.
    We add log(z) column-wise to the attention logits so larger tokens
    receive proportionally more attention weight.

        logits_compensated[:, :, :, k] += log(z_k)

    This is applied *inside* the MHSA scaled-dot-product, before softmax.

    Args:
        attention_scores : raw (unmasked, pre-softmax) scores, (B, H, Nq, Nk)
        token_sizes      : (B, Nk)  — current z values

    Returns:
        attention_scores with log-z bias added in-place-style (returns new tensor)
    """
    # log(z): (B, Nk) → broadcast to (B, 1, 1, Nk)
    log_z = torch.log(token_sizes.float().clamp(min=1.0))
    log_z = log_z[:, None, None, :]          # (B, 1, 1, Nk)
    return attention_scores + log_z

# Hook registration

def patch_token_fusion_hooks(
    vit_model: ViTModel,
    r: int = 16,
) -> List[BlockState]:
    """
    Install Token Fusion into all 12 ViT-Base encoder layers.

    For each ViTLayer in `vit_model.encoder.layer` this function:

      (a) MONKEY-PATCHES `ViTSelfAttention.forward` to:
          1. Compute head-averaged keys K̄ and store them in `state.keys`
             (Eq. 4a) before the attention computation begins.
          2. Inject the log-z column bias into QK^T/√d BEFORE softmax so
             that merged (larger) tokens receive proportionally more
             attention weight (Eq. 5).

      (b) REGISTERS a post-forward hook on `ViTLayer` (the whole block,
          i.e., after the FFN) that performs:
          - Alternating A/B partition of patch tokens (Eq. 3).
          - Bipartite soft matching to find top-r similar pairs (Eq. 4b).
          - Size-weighted token merge (Eq. 11 / 12).
          - Updates `state.token_sizes` and `state.keys` for the next block.

    Args:
        vit_model : A HuggingFace ViTModel (backbone, ideally frozen).
        r         : Merge budget — number of token pairs to remove per block.

    Returns:
        List of BlockState objects (one per layer), useful for inspection /
        resetting token_sizes between batches.
    """
    import math as _math

    states: List[BlockState] = []

    for layer_idx, vit_layer in enumerate(vit_model.encoder.layer):
        state = BlockState(r=r)
        states.append(state)

        # (a)  Monkey-patch ViTSelfAttention.forward
        #      — captures K̄ (Eq. 4a) and injects log-z bias (Eq. 5)

        # Bind the module reference at patch time
        self_attn_module = vit_layer.attention.attention

        def _make_patched(sa_mod, st):
            def patched(hidden_states, head_mask=None, output_attentions=False):
                import math as _m
                mixed_query = sa_mod.query(hidden_states)
                k_proj = sa_mod.key(hidden_states)
                v_proj = sa_mod.value(hidden_states)

                # Eq. 4a: head-averaged K̄
                B, N, D = hidden_states.shape
                H      = sa_mod.num_attention_heads
                d_h    = sa_mod.attention_head_size
                k_bar  = k_proj.view(B, N, H, d_h).mean(dim=2)   # (B, N, d_h)
                st.keys = k_bar[:, 1:, :].detach()                # (B, N_patch, d_h)

                key_layer   = sa_mod.transpose_for_scores(k_proj)
                val_layer   = sa_mod.transpose_for_scores(v_proj)
                qry_layer   = sa_mod.transpose_for_scores(mixed_query)

                attn_scores = torch.matmul(qry_layer, key_layer.transpose(-1, -2))
                attn_scores = attn_scores / _m.sqrt(d_h)

                # Eq. 5: log-z column-wise bias (only after first merge has occurred)
                if st.token_sizes is not None and st.token_sizes.shape[1] == N:
                    log_z = torch.log(st.token_sizes.float().clamp(min=1.0))
                    attn_scores = attn_scores + log_z[:, None, None, :]

                attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
                attn_probs = sa_mod.dropout(attn_probs)
                if head_mask is not None:
                    attn_probs = attn_probs * head_mask

                ctx = torch.matmul(attn_probs, val_layer)
                ctx = ctx.permute(0, 2, 1, 3).contiguous()
                ctx = ctx.view(ctx.size()[:-2] + (sa_mod.all_head_size,))

                return (ctx, attn_probs) if output_attentions else (ctx,)

            return patched

        # Replace the forward method of this layer's ViTSelfAttention
        self_attn_module.forward = _make_patched(self_attn_module, state)

        # (b)  Post-hook on ViTLayer — token merging after FFN (Eq. 3/4b/11/12)

        def make_layer_hook(st: BlockState, layer_i: int):
            def layer_post_hook(module, args, output):
                """
                output from ViTLayer.forward is a tuple:
                    (hidden_states,) or (hidden_states, attn_weights)
                hidden_states shape: (B, N_current, D)
                """
                hidden = output[0]          # (B, N_current, D)
                B, N_current, D = hidden.shape

                # Initialise token_sizes on the very first block
                if st.token_sizes is None or st.token_sizes.shape != (B, N_current):
                    st.token_sizes = torch.ones(
                        B, N_current, dtype=torch.float32, device=hidden.device
                    )

                # Keys must have been captured by the patched forward
                if st.keys is None:
                    return output

                N_patch = N_current - 1

                # Eq. 3: alternating A/B partition of patch tokens
                a_mask = torch.zeros(B, N_patch, dtype=torch.bool,
                                     device=hidden.device)
                a_mask[:, 0::2] = True   # even patch indices → set A

                keys_a = st.keys[:, 0::2, :]   # (B, nA, d_head)
                keys_b = st.keys[:, 1::2, :]   # (B, nB, d_head)

                nA, nB = keys_a.shape[1], keys_b.shape[1]
                if nA == 0 or nB == 0 or N_patch <= 1:
                    return output

                r_eff = min(st.r, nA, nB)
                if r_eff == 0:
                    return output

                # Eq. 4b — bipartite soft matching
                idx_a, idx_b = bipartite_soft_matching(keys_a, keys_b, r_eff)

                # Eq. 11 / 12 — size-weighted token merge
                hidden_new, keys_new, sizes_new = merge_tokens(
                    hidden, st.keys, st.token_sizes, idx_a, idx_b, a_mask
                )

                # Update state for the *next* block's attention
                st.token_sizes = sizes_new
                st.keys        = keys_new

                # Return modified sequence (preserving tuple structure)
                return (hidden_new,) + output[1:]

            return layer_post_hook

        vit_layer.register_forward_hook(make_layer_hook(state, layer_idx))

    return states

# Utility: reset token sizes between inference runs

def reset_token_sizes(states: List[BlockState]) -> None:
    """
    Clear per-sample state so that the next forward pass starts fresh.
    Must be called between batches when using persistent BlockState objects.
    (In practice the layer_post_hook re-initialises sizes on shape change,
    but explicit reset avoids stale state when batch size changes.)
    """
    for st in states:
        st.token_sizes = None
        st.keys        = None

# Quick sanity check

if __name__ == "__main__":
    from transformers import ViTModel

    print("Loading ViT-Base backbone …")
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit.eval()

    states = patch_token_fusion_hooks(vit, r=16)
    print(f"Registered Token Fusion hooks on {len(states)} ViT layers.")

    dummy = torch.randn(2, 3, 224, 224)
    pixel_vals = dummy  # In real code, pass through ViTFeatureExtractor first

    # ViTModel expects pixel_values
    with torch.no_grad():
        out = vit(pixel_values=dummy, output_attentions=False)

    final_hidden = out.last_hidden_state      # (B, N_remaining, 768)
    cls_out      = final_hidden[:, 0, :]      # (B, 768)
    print(f"CLS token shape after Token Fusion: {cls_out.shape}")
    print(f"Sequence length after 12 blocks of merging: {final_hidden.shape[1]} "
          f"(started at 197 = 1 CLS + 196 patches)")
    print(f"Token sizes of first sample at final layer: "
          f"{states[-1].token_sizes[0] if states[-1].token_sizes is not None else 'N/A'}")
