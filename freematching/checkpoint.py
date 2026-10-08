"""Read tensor-only public checkpoints and trusted legacy training snapshots."""
from pathlib import Path
import torch
from safetensors.torch import load_file


def load_checkpoint(path):
    path = Path(path)
    if path.suffix != '.safetensors':
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    tensors = load_file(str(path), device='cpu')
    groups = {'vis': {}, 'flow_decoder': {}, 'output_decoder': 'flow'}
    for name, tensor in tensors.items():
        prefix, sep, key = name.partition('.')
        if prefix not in ('vis', 'flow_decoder') or not sep:
            raise ValueError(f'Unexpected checkpoint tensor: {name}')
        groups[prefix][key] = tensor
    if not groups['vis'] or not groups['flow_decoder']:
        raise ValueError('Checkpoint must contain transformer and flow decoder weights')
    return groups
