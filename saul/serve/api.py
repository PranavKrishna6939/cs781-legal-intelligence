"""Offline `LLM.generate`. The engine is built around an already-loaded model or a checkpoint directory."""
from __future__ import annotations

import torch

from saul.serve.config import EngineConfig
from saul.serve.engine import LLMEngine
from saul.serve.finish import postprocess, specials_of
from saul.serve.sampling import MBRParams, SamplingParams
from saul.serve.scheduler import RequestOutput


class LLM:
    def __init__(self, model=None, tokenizer=None, checkpoint: str | None = None, tokenizer_path: str | None = None,
                 **config):
        if model is None:
            if not checkpoint or not tokenizer_path:
                raise ValueError("pass a model and tokenizer, or checkpoint and tokenizer_path")
            from transformers import PreTrainedTokenizerFast
            from saul.model.modeling_saul import SaulForCausalLM
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            model = SaulForCausalLM.from_pretrained(checkpoint).to(device).eval()
            if device == "cuda" or str(device).startswith("cuda"):
                model = model.to(torch.bfloat16)
        cfg = EngineConfig(**config) if config else EngineConfig()
        if cfg.device == "cuda" and not torch.cuda.is_available():
            cfg.device = "cpu"
            cfg.cuda_graphs = False
            cfg.dtype = "float32"
        self.engine = LLMEngine(model, tokenizer, cfg)
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompts: list[str], params: SamplingParams | None = None, use_tqdm: bool = False,
                 prepend_bos: bool = True) -> list[RequestOutput]:
        params = params or SamplingParams()
        if use_tqdm:
            try:
                from tqdm import tqdm
                # progress is reported by the engine's finished count; a bar over prompts is enough
                bar = tqdm(total=len(prompts), desc="generate")
            except Exception:
                bar = None
        else:
            bar = None
        out = self.engine.generate(prompts, params, prepend_bos=prepend_bos)
        if bar is not None:
            bar.update(len(prompts))
            bar.close()
        return out

    def stats(self) -> dict:
        return self.engine.stats()

    def close(self) -> None:
        self.engine.close()


def continuations(model, tokenizer, prompts: list[str], n_samples: int, max_new_tokens: int, target_words: int,
                  greedy: bool, device, prepend_bos: bool = True) -> list[list[str]]:
    """The shape `saul.decode.generate` returns: per prompt, `n_samples` strings."""
    dev = torch.device(device)
    # size the pools to this call
    encoded = []
    bos = getattr(tokenizer, "bos_token_id", None)
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if prepend_bos and bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        encoded.append(ids or [bos or 0])
    max_prompt = max(8, max(len(x) for x in encoded))
    block = 16 if n_samples <= 16 else n_samples
    groups = max(1, min(len(prompts), 8))
    dtype = "float32" if dev.type == "cpu" else "bfloat16"
    cfg = dict(
        max_slots=block * groups, block_size=block, max_prompt_len=max_prompt, max_new_tokens=max_new_tokens,
        prefill_token_budget=max(max_prompt * groups, max_prompt), cuda_graphs=dev.type == "cuda",
        device=str(dev), dtype=dtype, rouge_workers=1, seed=0,
    )
    llm = LLM(model, tokenizer, **cfg)
    params = SamplingParams(
        n=n_samples, temperature=0.0 if greedy else 0.8, truncation="none" if greedy else "min_p",
        target_words=target_words, max_new_tokens=max_new_tokens, mbr=MBRParams(enabled=False),
    )
    outs = llm.generate(prompts, params, prepend_bos=prepend_bos)
    specials = specials_of(tokenizer)
    return [[postprocess(c, target_words, specials) for c in o.candidates] for o in outs]
