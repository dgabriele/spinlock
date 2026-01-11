#!/usr/bin/env python3
"""Debug script to test token-conditioned model loading."""

import torch
import yaml
from spinlock.noa.backbone import NOABackbone

checkpoint_path = "checkpoints/experiments/phase2/exp2b_token_baseline/meta_operator_best.pt"
config_path = "configs/noa/experiments/phase2/exp2b_token_baseline.yaml"

# Load config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']

# Detect tokens
uses_tokens = any(k.startswith('token_embedding.') for k in state_dict.keys())
print(f"Uses tokens: {uses_tokens}")

# Extract codebook sizes
codebook_sizes = []
idx = 0
while f'token_embedding.embeddings.{idx}.weight' in state_dict:
    emb_weight = state_dict[f'token_embedding.embeddings.{idx}.weight']
    codebook_size = emb_weight.shape[0]
    codebook_sizes.append(codebook_size)
    idx += 1

print(f"Codebook sizes: {codebook_sizes}")
print(f"Num tokens: {len(codebook_sizes)}")

# Create model
model_config = config['model']
model = NOABackbone(
    spatial_dim=model_config['spatial_dim'],
    in_channels=model_config['in_channels'],
    out_channels=model_config['out_channels'],
    base_channels=model_config['base_channels'],
    encoder_levels=model_config['encoder_levels'],
    modes=model_config['modes'],
    afno_blocks=model_config['afno_blocks'],
    token_conditioning=True,
    token_embed_dim=model_config.get('token_embed_dim', 64),
    num_tokens=len(codebook_sizes),
    codebook_sizes=codebook_sizes,
)

print(f"\nModel attributes after init:")
print(f"  token_conditioning: {model.token_conditioning}")
print(f"  _in_channels: {model._in_channels}")
print(f"  Has token_embedding: {hasattr(model, 'token_embedding')}")

# Load state dict
model.load_state_dict(state_dict)

print(f"\nModel attributes after load_state_dict:")
print(f"  token_conditioning: {model.token_conditioning}")
print(f"  _in_channels: {model._in_channels}")
print(f"  Has token_embedding: {hasattr(model, 'token_embedding')}")

# Check operator input channels
first_conv = model.operator.encoder.stem.block[0]
print(f"\nOperator first conv:")
print(f"  Weight shape: {first_conv.weight.shape}")
print(f"  Expected input channels: {first_conv.in_channels}")

# Test forward pass
print(f"\nTesting forward pass...")
ic = torch.randn(1, 1, 64, 64)
tokens = torch.randint(0, 20, (1, 36))

print(f"  IC shape: {ic.shape}")
print(f"  Tokens shape: {tokens.shape}")

try:
    with torch.no_grad():
        output = model.rollout(ic, steps=2, return_all_steps=True, tokens=tokens)
    print(f"  ✓ Success! Output shape: {output.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")

    # Debug: check what's being passed to operator
    print(f"\n  Debugging token embedding...")
    if model.token_conditioning and hasattr(model, 'token_embedding'):
        token_embed = model.token_embedding(tokens)
        print(f"    Token embed shape: {token_embed.shape}")

        B, C, H, W = ic.shape
        token_spatial = token_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)
        print(f"    Token spatial shape: {token_spatial.shape}")

        augmented = torch.cat([ic, token_spatial], dim=1)
        print(f"    Augmented input shape: {augmented.shape}")

        print(f"\n    Trying direct operator call with augmented input...")
        try:
            out = model.operator(augmented)
            print(f"    ✓ Direct operator call works! Output shape: {out.shape}")
        except Exception as e2:
            print(f"    ✗ Direct operator call failed: {e2}")
