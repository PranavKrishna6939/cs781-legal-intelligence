#!/usr/bin/env python
"""Export a saved trainer EMA on CPU without changing the source checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--expected-step', required=True, type=int)
    args = parser.parse_args()
    source, dest = args.checkpoint.resolve(), args.out.resolve()
    assert source != dest and not dest.exists(), 'Use a new output directory'
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    state = torch.load(source / 'trainer.pt', map_location='cpu', mmap=True,
                       weights_only=False)
    assert state['step'] == args.expected_step
    ema = state['ema']
    assert ema['updates'] > 0 and ema['shadow']
    shadow = ema['shadow']
    with safe_open(source / 'model.safetensors', framework='pt', device='cpu') as raw:
        # These checkpoints save one tensor per named parameter, with tied
        # embeddings represented once. Reject a different layout explicitly.
        assert set(raw.keys()) == set(shadow), 'Raw/EMA parameter keys differ'
        metadata = raw.metadata()
        for name, tensor in shadow.items():
            assert list(tensor.shape) == raw.get_slice(name).get_shape(), name
            assert tensor.dtype == torch.float32 and tensor.device.type == 'cpu', name
            assert tensor.is_contiguous() and torch.isfinite(tensor).all(), name
    temporary = dest.with_name(dest.name + f'.tmp-{os.getpid()}')
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        save_file(shadow, temporary / 'model.safetensors', metadata=metadata)
        shutil.copyfile(source / 'config.json', temporary / 'config.json')
        with safe_open(temporary / 'model.safetensors', framework='pt', device='cpu') as saved:
            assert set(saved.keys()) == set(shadow)
            for name, tensor in shadow.items():
                assert torch.equal(saved.get_tensor(name), tensor), name
        manifest = dict(
            source_checkpoint=str(source), step=state['step'],
            ema_updates=ema['updates'], ema_decay=ema['decay'],
            parameter_tensors=len(shadow), parameters=sum(t.numel() for t in shadow.values()),
            trainer_sha256=sha256(source / 'trainer.pt'),
            source_model_sha256=sha256(source / 'model.safetensors'),
            config_sha256=sha256(temporary / 'config.json'),
            model_sha256=sha256(temporary / 'model.safetensors'),
            verification='All saved tensors exactly equal CPU EMA shadow; finite FP32; matching raw keys/shapes',
            cuda_initialized=torch.cuda.is_initialized())
        assert not manifest['cuda_initialized']
        (temporary / 'ema_export.json').write_text(json.dumps(manifest, indent=2) + '\n')
        assert not dest.exists()
        temporary.rename(dest)
        print(json.dumps(dict(output=str(dest), **manifest)), flush=True)
    except BaseException:
        shutil.rmtree(temporary)
        raise


if __name__ == '__main__':
    main()
