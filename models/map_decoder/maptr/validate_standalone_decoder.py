"""
Validation script: verifies that StandaloneMapTRDecoder produces identical
outputs to the original mmcv/mmdet-based MapTRDecoder.

Run from the projects/ directory:
    python validate_standalone_decoder.py

What this script does:
  1. Builds the original MapTRDecoder via mmcv's registry (same config as
     maptr_tiny_r50_24e.py).
  2. Builds the standalone re-implementation (pure PyTorch).
  3. Copies all weights from original → standalone with load_state_dict().
  4. Feeds identical random inputs to both in eval mode (no dropout).
  5. Asserts that outputs and reference-point tensors match to < 1e-5.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Make sure "projects/" is on the Python path so mmdet3d_plugin is importable
# ---------------------------------------------------------------------------
PROJECTS_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECTS_DIR not in sys.path:
    sys.path.insert(0, PROJECTS_DIR)

# Also need the workspace root for mmdet3d / mmcv resolution
WORKSPACE_ROOT = os.path.dirname(PROJECTS_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import importlib.util
import torch
import torch.nn as nn


def _stub_package(pkg_name: str):
    """Insert an empty module stub so Python won't execute the real __init__.py."""
    import types
    if pkg_name not in sys.modules:
        mod = types.ModuleType(pkg_name)
        mod.__path__ = []  # mark as package
        sys.modules[pkg_name] = mod


def _import_file(module_name: str, file_path: str):
    """Import a single .py file and register it under module_name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Import original (mmcv/mmdet-based) modules
# We stub out the heavy mmdet3d_plugin package __init__ (which transitively
# imports IPython → sqlite3 → CXXABI crash) and import only the files we need.
# ---------------------------------------------------------------------------
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

# 1) Register DetrTransformerDecoderLayer (mmdet)
import mmdet.models.utils.transformer  # noqa: F401

# 2) Stub every package layer that has a problematic __init__.py
for _pkg in [
    "mmdet3d_plugin",
    "mmdet3d_plugin.bevformer",
    "mmdet3d_plugin.bevformer.modules",
    "mmdet3d_plugin.maptr",
    "mmdet3d_plugin.maptr.modules",
]:
    _stub_package(_pkg)

# 3) Import bevformer multi_scale_deformable_attn_function (needed by decoder)
_import_file(
    "mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function",
    os.path.join(
        PROJECTS_DIR,
        "mmdet3d_plugin", "bevformer", "modules",
        "multi_scale_deformable_attn_function.py",
    ),
)

# 4) Register CustomMSDeformableAttention
_import_file(
    "mmdet3d_plugin.bevformer.modules.decoder",
    os.path.join(
        PROJECTS_DIR,
        "mmdet3d_plugin", "bevformer", "modules", "decoder.py",
    ),
)

# 5) Register MapTRDecoder
_import_file(
    "mmdet3d_plugin.maptr.modules.decoder",
    os.path.join(
        PROJECTS_DIR,
        "mmdet3d_plugin", "maptr", "modules", "decoder.py",
    ),
)

# ---------------------------------------------------------------------------
# Import standalone implementation
# ---------------------------------------------------------------------------
from standalone_maptr_decoder import StandaloneMapTRDecoder


# ---------------------------------------------------------------------------
# Config (matches maptr_tiny_r50_24e.py decoder section)
# ---------------------------------------------------------------------------
EMBED_DIMS = 256
FFN_DIMS   = 512
NUM_HEADS  = 8
NUM_LAYERS = 6
NUM_LEVELS = 1
NUM_POINTS = 4
ATTN_DROP  = 0.1
FFN_DROP   = 0.1

ORIGINAL_CFG = dict(
    type="MapTRDecoder",
    num_layers=NUM_LAYERS,
    return_intermediate=True,
    transformerlayers=dict(
        type="DetrTransformerDecoderLayer",
        attn_cfgs=[
            dict(
                type="MultiheadAttention",
                embed_dims=EMBED_DIMS,
                num_heads=NUM_HEADS,
                dropout=ATTN_DROP,
            ),
            dict(
                type="CustomMSDeformableAttention",
                embed_dims=EMBED_DIMS,
                num_levels=NUM_LEVELS,
            ),
        ],
        feedforward_channels=FFN_DIMS,
        ffn_dropout=FFN_DROP,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_original() -> nn.Module:
    return build_transformer_layer_sequence(ORIGINAL_CFG)


def build_standalone() -> StandaloneMapTRDecoder:
    return StandaloneMapTRDecoder(
        num_layers=NUM_LAYERS,
        embed_dims=EMBED_DIMS,
        num_heads=NUM_HEADS,
        ffn_channels=FFN_DIMS,
        ffn_drop=FFN_DROP,
        attn_drop=ATTN_DROP,
        num_levels=NUM_LEVELS,
        num_points=NUM_POINTS,
        return_intermediate=True,
    )


def make_inputs(bs=1, num_query=50, bev_h=30, bev_w=60, seed=42):
    """Return a fixed set of random inputs that both decoders accept."""
    torch.manual_seed(seed)
    num_bev = bev_h * bev_w
    query            = torch.randn(num_query, bs, EMBED_DIMS)
    query_pos        = torch.randn(num_query, bs, EMBED_DIMS)
    bev_embed        = torch.randn(num_bev, bs, EMBED_DIMS)
    reference_points = torch.sigmoid(torch.randn(bs, num_query, 2))
    spatial_shapes   = torch.tensor([[bev_h, bev_w]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)
    return dict(
        query=query,
        query_pos=query_pos,
        bev_embed=bev_embed,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(verbose: bool = True):
    print("=" * 60)
    print("Building original MapTRDecoder …")
    original   = build_original().eval()
    print("Building StandaloneMapTRDecoder …")
    standalone = build_standalone().eval()

    # ----- weight transfer -----
    orig_sd = original.state_dict()
    stan_sd = standalone.state_dict()

    missing   = [k for k in orig_sd if k not in stan_sd]
    extra     = [k for k in stan_sd if k not in orig_sd]
    if missing:
        print(f"[WARN] Keys in original but NOT in standalone ({len(missing)}):")
        for k in missing:
            print(f"       {k}")
    if extra:
        print(f"[WARN] Keys in standalone but NOT in original ({len(extra)}):")
        for k in extra:
            print(f"       {k}")
    if not missing and not extra:
        print("State-dict keys match perfectly.")

    standalone.load_state_dict(orig_sd, strict=True)
    print("Weights copied: original → standalone (strict=True)\n")

    # ----- forward pass -----
    inp = make_inputs()
    common_kwargs = dict(
        spatial_shapes=inp["spatial_shapes"],
        level_start_index=inp["level_start_index"],
    )

    with torch.no_grad():
        out_orig, ref_orig = original(
            query=inp["query"],
            key=None,
            value=inp["bev_embed"],
            query_pos=inp["query_pos"],
            reference_points=inp["reference_points"],
            reg_branches=None,
            key_padding_mask=None,
            **common_kwargs,
        )
        out_stan, ref_stan = standalone(
            query=inp["query"],
            key=None,
            value=inp["bev_embed"],
            query_pos=inp["query_pos"],
            reference_points=inp["reference_points"],
            reg_branches=None,
            key_padding_mask=None,
            **common_kwargs,
        )

    # ----- comparison -----
    print(f"Output shape          : {tuple(out_orig.shape)}")
    print(f"Reference-pts shape   : {tuple(ref_orig.shape)}")

    diff_out = (out_orig - out_stan).abs()
    diff_ref = (ref_orig - ref_stan).abs()

    print(f"\n{'':30s}  {'max':>12s}  {'mean':>12s}")
    print(f"{'output diff':30s}  {diff_out.max().item():12.2e}  {diff_out.mean().item():12.2e}")
    print(f"{'reference_points diff':30s}  {diff_ref.max().item():12.2e}  {diff_ref.mean().item():12.2e}")

    THRESHOLD = 1e-5
    assert diff_out.max().item() < THRESHOLD, (
        f"Output mismatch: max diff = {diff_out.max().item():.2e} (threshold {THRESHOLD})"
    )
    assert diff_ref.max().item() < THRESHOLD, (
        f"Reference-points mismatch: max diff = {diff_ref.max().item():.2e} (threshold {THRESHOLD})"
    )

    print(f"\n✓  All outputs match (max diff < {THRESHOLD})")
    print("=" * 60)


if __name__ == "__main__":
    validate()
