"""FastAPI front end. One engine thread; the event loop only enqueues and waits."""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from saul.serve.sampling import MBRParams, SamplingParams


def _params(data: dict | None) -> SamplingParams:
    data = dict(data or {})
    mbr = data.pop("mbr", None)
    if isinstance(mbr, dict):
        data["mbr"] = MBRParams(**mbr)
    # JSON nulls
    nullable = {"target_words", "seed", "forced_tokens"}
    data = {k: v for k, v in data.items() if v is not None or k in nullable}
    if "stop_token_ids" in data and data["stop_token_ids"] is not None:
        data["stop_token_ids"] = tuple(data["stop_token_ids"])
    if "forced_tokens" in data and data["forced_tokens"] is not None:
        data["forced_tokens"] = tuple(data["forced_tokens"])
    return SamplingParams(**data)


def build_app(llm, max_queue: int = 1024):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    engine = llm.engine
    engine.sched.max_waiting = max_queue
    lock = threading.Lock()
    app.state.llm = llm

    def _run_prompts(prompts: list[str], params: SamplingParams, rng_seed: int | None = None):
        with lock:
            if rng_seed is not None:
                import torch
                torch.manual_seed(rng_seed)
            return engine.generate(prompts, params)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/stats")
    def stats():
        return engine.stats()

    @app.get("/v1/models")
    def models():
        cfg = engine.model.config
        return {"arch": cfg.arch, "n_layers": cfg.n_layers, "d_model": cfg.d_model, "vocab": cfg.vocab_size}

    @app.post("/v1/generate")
    async def generate(body: dict):
        prompts = body.get("prompts")
        if prompts is None and "prompt" in body:
            prompts = [body["prompt"]]
        if not prompts or not isinstance(prompts, list):
            raise HTTPException(400, "prompt or prompts is required")
        try:
            params = _params(body.get("sampling"))
            rng_seed = body.get("rng_seed")
            if rng_seed is not None and (type(rng_seed) is not int or not 0 <= rng_seed < 2**63):
                raise ValueError("rng_seed must be an integer in [0, 2**63)")
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if len(engine.sched.queue) + len(prompts) > max_queue:
            raise HTTPException(429, "queue is full")
        loop = asyncio.get_running_loop()
        try:
            outs = await loop.run_in_executor(None, _run_prompts, prompts, params, rng_seed)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except BufferError as exc:
            raise HTTPException(429, str(exc)) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception("Generation request failed")
            raise HTTPException(500, str(exc)) from exc
        payload: dict[str, Any] = {"outputs": []}
        for o in outs:
            item = {"text": o.text, "finish_reasons": o.finish_reasons, "prompt_tokens": o.prompt_tokens,
                    "gen_tokens": o.gen_tokens}
            if body.get("return_candidates"):
                item["candidates"] = o.candidates
            payload["outputs"].append(item)
        return JSONResponse(payload)

    @app.post("/v1/generate_batch")
    async def generate_batch(body: dict):
        prompts = body.get("prompts") or []
        if not isinstance(prompts, list) or not prompts:
            raise HTTPException(400, "prompts is required")
        try:
            params = _params(body.get("sampling"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

        async def stream():
            import json
            loop = asyncio.get_running_loop()
            try:
                outs = await loop.run_in_executor(None, _run_prompts, prompts, params)
            except Exception as exc:
                yield json.dumps({"error": str(exc)}) + "\n"
                return
            for i, o in enumerate(outs):
                yield json.dumps({"i": i, "text": o.text, "finish_reasons": o.finish_reasons}) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/v1/rerank_features")
    async def rerank_features(body: dict):
        groups=body.get('groups')
        if not isinstance(groups,list) or not 1<=len(groups)<=32:
            raise HTTPException(400,'groups must contain 1..32 prompt/candidate groups')

        def compute():
            from saul.serve.rerank import contextual_features
            with lock:
                return [dict(id=g['id'],**contextual_features(
                    engine.model,engine.tok,g['prompt'],g['candidates'],
                    context_tokens=int(body.get('context_tokens',1024)))) for g in groups]
        try:
            out=await asyncio.get_running_loop().run_in_executor(None,compute)
        except (KeyError,TypeError,ValueError) as exc:
            raise HTTPException(400,str(exc)) from exc
        return {'groups':out}

    # Research features share the generation lock: no concurrent mutation of
    # attention configuration or CUDA model state within this server.
    latent_bank = None

    @app.post("/v1/latent_bank")
    async def set_latent_bank(body: dict):
        def compute():
            nonlocal latent_bank
            from saul.serve.latent import make_bank
            with lock:
                latent_bank = make_bank(body, engine.model.config.d_model,
                                        next(engine.model.parameters()).device)
                return {'bank_id': latent_bank['id']}
        try:
            return await asyncio.get_running_loop().run_in_executor(None, compute)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/latent_features")
    async def get_latent_features(body: dict):
        def compute():
            from saul.serve.latent import latent_features
            with lock:
                bank = None
                if body.get('bank_id') is not None:
                    if latent_bank is None or body['bank_id'] != latent_bank['id']:
                        raise ValueError('Requested bank is not installed')
                    bank = latent_bank
                return {'features': latent_features(engine.model, engine.tok, body['texts'],
                    kind=body.get('kind', 'future'), sample_tokens=int(body.get('sample_tokens', 0)), bank=bank)}
        try:
            return await asyncio.get_running_loop().run_in_executor(None, compute)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/mbr_features")
    async def get_mbr_features(body: dict):
        def compute():
            import numpy as np
            from saul.serve.finish import filter_candidates, embed_groups, rouge_groups, postprocess, specials_of
            from saul.decode.mbr import mbr_pick
            texts, reasons = body['candidates'], body['finish_reasons']
            if not isinstance(texts, list) or not 1 <= len(texts) <= 64 or len(texts) != len(reasons):
                raise ValueError('Need 1..64 candidates with matching finish reasons')
            if any(not isinstance(t, str) or len(t) > 100000 for t in texts):
                raise ValueError('Invalid candidate text')
            kept = filter_candidates(texts, reasons, 128)
            with lock:
                embed = embed_groups(engine.model, engine.tok, [kept], next(engine.model.parameters()).device)[0]
                rouge = rouge_groups([kept], 1)[0]
                matrix = .5 * embed + .5 * rouge
                scores = (matrix.sum(1) - np.diag(matrix)) / max(1, len(kept)-1)
                selected = postprocess(kept[mbr_pick(matrix)], 128, specials_of(engine.tok))
            return {'candidates': kept, 'scores': scores.tolist(), 'selected': selected}
        try:
            return await asyncio.get_running_loop().run_in_executor(None, compute)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/layer_mbr_features")
    async def get_layer_mbr_features(body: dict):
        def compute():
            from saul.serve.layer_mbr import layer_mbr_features
            with lock:
                return layer_mbr_features(engine.model, engine.tok, body['candidates'],
                                          body['finish_reasons'], body['layers'])
        try:
            return await asyncio.get_running_loop().run_in_executor(None, compute)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/large_layer_mbr_features")
    async def get_large_layer_mbr_features(body: dict):
        def compute():
            from saul.serve.large_layer_mbr import large_layer_mbr_features
            with lock:
                return large_layer_mbr_features(engine.model, engine.tok, body['candidates'],
                                                body['finish_reasons'], body['layers'])
        try:
            return await asyncio.get_running_loop().run_in_executor(None, compute)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/large_mbr_features")
    async def get_large_mbr_features(body: dict):
        def compute():
            from saul.serve.large_mbr import large_mbr_features
            with lock:
                return large_mbr_features(engine.model, engine.tok, body['candidates'], body['finish_reasons'],
                                          prompt=body.get('prompt'))
        try:
            return await asyncio.get_running_loop().run_in_executor(None, compute)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    return app
