"""CPU tests for the inference engine: logit parity, reuse, sampling, stopping, MBR, API."""
from __future__ import annotations

import torch

from saul.decode.mbr import mbr_choose
from saul.decode.processors import EtaSampling, GeneratedNoRepeatNGram, MinP, SuppressTokens, TopP, WindowedFrequencyPenalty
from saul.model.modeling_saul import build_model, tiny_config
from saul.serve.config import EngineConfig
from saul.serve.engine import LLMEngine
from saul.serve.finish import filter_candidates
from saul.serve.sampling import NEG, SamplingParams, apply_penalties, ngram_ban_dense
from saul.serve.words import WordCounter


class Tok:
    """Enough of a tokenizer for the engine: ids decode to `w{id} ` so each token is one finished word."""

    def __init__(self, vocab=128):
        self.vocab = vocab
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.all_special_ids = [0, 1, 2]
        self.all_special_tokens = ["<pad>", "<bos>", "<eos>"]
        self.added_tokens_decoder = {}

    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        parts = text.split()
        if parts and parts[0].startswith("w") and parts[0][1:].isdigit():
            return [int(p[1:]) for p in parts]
        return [int(x) for x in parts]

    def decode(self, ids, **kwargs):
        return "".join(f"w{int(i)} " for i in ids)

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [self.encode(t) for t in texts]}


def _model(arch):
    torch.manual_seed(0)
    model = build_model(tiny_config(arch), enforce_min=False).eval()
    for p in model.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.05)
    return model


def _engine(model, slots=8, block=2, prompt=64, new=16):
    cfg = EngineConfig(
        max_slots=slots, block_size=block, max_prompt_len=prompt, max_new_tokens=new,
        prefill_token_budget=max(256, prompt), cuda_graphs=False, device="cpu", dtype="float32",
        suffix_buckets=(4, 8, new), prompt_buckets=(16, 32, prompt), slot_buckets=None,
        rouge_workers=1, seed=0, mbr_batch_groups=2,
    )
    return LLMEngine(model, Tok(), cfg)


def _run(engine, prompts, forced, ns, max_news):
    params = []
    for n, m, f in zip(ns, max_news, forced):
        params.append(SamplingParams(
            n=n, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
            target_words=None, max_new_tokens=m, forced_tokens=tuple(f), record_logits=True,
        ))
    reqs = [engine.submit_ids(p, par) for p, par in zip(prompts, params)]
    engine.run_until_complete()
    return reqs


def test_step_parity_all_architectures_with_churn():
    for arch in ("dense", "hybrid", "ced", "conv"):
        model = _model(arch)
        eng = _engine(model)
        torch.manual_seed(1)
        v = model.config.vocab_size
        lengths = [3, 8, 17, 5, 11]
        prompts = [torch.randint(4, v, (n,)).tolist() for n in lengths]
        ns = [1, 2, 1, 2, 1]
        news = [2, 5, 3, 4, 6]
        forced = [torch.randint(4, v, (m,)).tolist() for m in news]
        reqs = _run(eng, prompts, forced, ns, news)
        for req, prompt, force in zip(reqs, prompts, forced):
            assert req.output is not None, arch
            full = model(torch.tensor([prompt + force])).logits[0]
            for local, trace in enumerate(req.traces):
                assert len(trace) == len(force), (arch, local, len(trace), len(force))
                for t, row in enumerate(trace):
                    ref = full[len(prompt) - 1 + t]
                    if not torch.allclose(row, ref, atol=1e-4):
                        raise AssertionError(f"{arch} slot {local} step {t} max {(row - ref).abs().max().item()}")


def test_slot_reuse_matches_a_fresh_engine():
    model = _model("hybrid")
    v = model.config.vocab_size
    prompt = torch.randint(4, v, (9,)).tolist()
    forced = torch.randint(4, v, (4,)).tolist()
    long_p = torch.randint(4, v, (15,)).tolist()
    long_f = torch.randint(4, v, (6,)).tolist()
    first = _engine(model)
    # a long occupant, then the prompt we care about reuses its block
    _run(first, [long_p, prompt], [long_f, forced], [2, 1], [6, 4])
    # the second request is the one we compare; run it alone on a fresh engine
    fresh = _engine(model)
    a = _run(first, [prompt], [forced], [1], [4])[0]
    b = _run(fresh, [prompt], [forced], [1], [4])[0]
    for x, y in zip(a.traces[0], b.traces[0]):
        assert torch.allclose(x, y, atol=1e-5)


def test_packed_prefill_matches_single_prompts():
    model = _model("ced")
    eng = _engine(model, slots=4, block=1, prompt=32, new=4)
    v = model.config.vocab_size
    prompts = [torch.randint(4, v, (6,)).tolist(), torch.randint(4, v, (10,)).tolist()]
    # one packed prefill via the runner, compared with two singleton prefills
    jobs = [{"g": i, "n": 1, "ids": p} for i, p in enumerate(prompts)]
    packed = eng.runner.prefill(jobs)
    for i, p in enumerate(prompts):
        alone_eng = _engine(model, slots=1, block=1, prompt=32, new=4)
        alone = alone_eng.runner.prefill([{"g": 0, "n": 1, "ids": p}])
        assert torch.allclose(packed[i], alone[0], atol=1e-4)
        ref = model(torch.tensor([p])).logits[0, -1]
        assert torch.allclose(packed[i], ref, atol=1e-4)


def test_sampler_matches_processors():
    torch.manual_seed(0)
    rows, length, vocab = 4, 12, 50
    generated = torch.randint(0, vocab, (rows, length))
    logits = torch.randn(rows, vocab)
    for n in (2, 3, 4):
        a = GeneratedNoRepeatNGram(n)(logits.clone(), generated)
        b = ngram_ban_dense(logits.clone(), generated, n)
        assert torch.equal(a <= NEG / 2, b <= NEG / 2)
    # frequency
    freq = WindowedFrequencyPenalty(5, 0.4)(logits.clone(), generated)
    # rebuild counts the engine's way and compare the penalty
    from saul.serve.state import StatePools
    spec = {"ng": ("slot_seq", (1,), None)}
    pools = StatePools([spec], None, 1, rows, 4, length, vocab, "cpu", torch.float32)
    pools.slen[:] = length
    pools.out_tokens[:, :length] = generated
    pools.freq_pen[:] = 0.4
    pools.freq_win[:] = 5
    pools.ngram_n[:] = 0
    # fill freq the way update_freq would have, by brute force
    for s in range(rows):
        tail = generated[s, -5:]
        for t in tail.tolist():
            pools.freq[s, t] += 1
    got = apply_penalties(logits.clone(), pools, torch.arange(rows), None, levels=())
    assert torch.allclose(got, freq, atol=1e-5)
    base = logits.clone()
    assert torch.equal(MinP(0.1)(base.clone(), generated) < -1e8, _trunc_mask(base, "min_p", 0.1))
    assert torch.equal(TopP(0.4)(base.clone(), generated) < -1e8, _trunc_mask(base, "top_p", 0.4))
    assert torch.equal(EtaSampling(1e-3)(base.clone(), generated) < -1e8, _trunc_mask(base, "eta", 1e-3))
    banned = SuppressTokens([1, 2, 7])(base.clone(), generated)
    assert (banned[:, [1, 2, 7]] < -1e8).all()


def _trunc_mask(logits, kind, value):
    from saul.serve.sampling import _truncate
    k = {"min_p": 1, "top_p": 2, "eta": 3}[kind]
    kind_t = torch.full((logits.shape[0],), k, dtype=torch.int8)
    min_p = torch.full((logits.shape[0],), value if kind == "min_p" else 0.05)
    top_p = torch.full((logits.shape[0],), value if kind == "top_p" else 0.9)
    eta = torch.full((logits.shape[0],), value if kind == "eta" else 3e-3)
    out = _truncate(logits, kind_t, min_p, top_p, eta, do_top_p=(kind == "top_p"), do_eta=(kind == "eta"))
    return out < -1e8


def test_stopping_completes_the_last_word_and_reports_length_and_stop():
    model = _model("dense")
    eng = _engine(model, slots=4, block=1, prompt=16, new=32)
    prompt = [4, 5, 6, 7]
    # greedy, no penalties: run well past 4 words, and a short target
    long = SamplingParams(n=1, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
                          target_words=None, max_new_tokens=12)
    short = SamplingParams(n=1, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
                           target_words=4, max_new_tokens=12)
    a = eng.submit_ids(prompt, long)
    eng.run_until_complete()
    eng2 = _engine(model, slots=2, block=1, prompt=16, new=32)
    b = eng2.submit_ids(prompt, short)
    eng2.run_until_complete()
    long_words = a.output.candidates[0].split()
    short_words = b.output.text.split()
    assert len(short_words) == 4
    assert short_words[-1] == long_words[3]
    # length cap
    eng3 = _engine(model, slots=1, block=1, prompt=16, new=5)
    c = eng3.submit_ids(prompt, SamplingParams(n=1, temperature=0.0, truncation="none", no_repeat_ngram=0,
                                               frequency_penalty=0.0, target_words=None, max_new_tokens=5))
    eng3.run_until_complete()
    assert c.output.finish_reasons == ["length"]
    assert c.output.gen_tokens == [5]
    # stop id
    eng4 = _engine(model, slots=1, block=1, prompt=16, new=8)
    d = eng4.submit_ids(prompt, SamplingParams(
        n=1, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
        target_words=None, max_new_tokens=8, forced_tokens=(8, 9, 2), stop_token_ids=(2,)))
    eng4.run_until_complete()
    assert d.output.finish_reasons == ["stop"]
    assert 2 not in eng4.pools.out_tokens[0, :d.output.gen_tokens[0]].tolist() or d.output.gen_tokens[0] <= 2


def test_mbr_picks_the_same_candidate_as_mbr_choose():
    model = _model("dense")
    tok = Tok()
    eng = _engine(model)
    cands = ["w1 w2 w3", "w1 w2 w9", "w8 w8 w8"]
    # the finisher sees raw generations; rouge utility must match mbr_choose
    groups = [{"texts": cands, "reasons": ["words"] * 3, "target": 3, "utility": "rouge", "alpha": 0.5, "mbr": True}]
    got = eng.finisher.choose(groups)[0]
    assert got.split() == mbr_choose(cands, "rouge").split()
    groups[0]["utility"] = "embed"
    got_e = eng.finisher.choose(groups)[0]
    assert got_e.split() == mbr_choose(cands, "embed", model=model, tokenizer=tok, device="cpu").split()
    assert filter_candidates(["w1", "w1 w2"], ["length", "words"], 2) == ["w1 w2"]
    assert filter_candidates(["onlyone"], ["length"], 4) == ["onlyone"]


def test_greedy_is_deterministic_across_batch_shapes():
    model = _model("dense")
    v = model.config.vocab_size
    prompt = torch.randint(4, v, (7,)).tolist()
    params = SamplingParams(n=1, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
                            target_words=3, max_new_tokens=8)
    a = _engine(model, slots=2, block=1)
    b = _engine(model, slots=4, block=2)
    ra = a.submit_ids(prompt, params)
    # a second, different prompt in the same batch must not change the greedy tokens
    other = torch.randint(4, v, (4,)).tolist()
    a.submit_ids(other, params)
    a.run_until_complete()
    rb = b.submit_ids(list(prompt), params)
    b.run_until_complete()
    assert ra.output.candidates[0] == rb.output.candidates[0]


def test_generate_preserves_order_and_server_smoke():
    model = _model("conv")
    tok = Tok()
    cfg = dict(max_slots=4, block_size=1, max_prompt_len=16, max_new_tokens=4, prefill_token_budget=64,
               cuda_graphs=False, device="cpu", dtype="float32", rouge_workers=1)
    from saul.serve.api import LLM
    llm = LLM(model, tok, **cfg)
    # prompts are space-separated ids; bos is prepended by default and Tok.bos is 1, which the encode path
    # does not understand as text. Submit via the engine to keep the test about order.
    outs = llm.engine.generate(["4 5 6", "7 8"], SamplingParams(
        n=1, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
        target_words=None, max_new_tokens=3), prepend_bos=False)
    assert len(outs) == 2 and all(o.gen_tokens == [3] for o in outs)
    try:
        from fastapi.testclient import TestClient
        from saul.serve.server import build_app
    except Exception:
        return
    app = build_app(llm, max_queue=2)
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["ok"]
    bad = client.post("/v1/generate", json={"prompt": "4 5", "sampling": {"temperature": -1}})
    assert bad.status_code == 400
    ok = client.post("/v1/generate", json={"prompt": "4 5 6", "sampling": {
        "temperature": 0, "n": 1, "max_new_tokens": 2, "target_words": None, "no_repeat_ngram": 0,
        "frequency_penalty": 0}})
    assert ok.status_code == 200 and "text" in ok.json()["outputs"][0]
    seeded = {"prompt": "4 5 6", "rng_seed": 781, "return_candidates": True,
              "sampling": {"temperature": 0.9, "n": 1, "max_new_tokens": 4,
                           "target_words": None, "no_repeat_ngram": 0, "frequency_penalty": 0}}
    first = client.post("/v1/generate", json=seeded)
    client.post("/v1/generate", json={**seeded, "rng_seed": 982})
    repeat = client.post("/v1/generate", json=seeded)
    assert first.status_code == repeat.status_code == 200
    assert first.json()["outputs"] == repeat.json()["outputs"]
    assert client.post("/v1/generate", json={**seeded, "rng_seed": "781"}).status_code == 400


def test_ngram_occurrence_limit_matches_counter_and_slot_reuse():
    from collections import Counter
    from saul.serve.sampling import ngram_ban_pool, store_ngram_keys
    from saul.serve.state import StatePools
    pools = StatePools([], None, 1, 2, 16, 32, 128, 'cpu', torch.float32)
    pools.ngram_n[:] = 4
    pools.ngram_max[:] = torch.tensor([1, 2])
    idx = torch.arange(2)
    seq = [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6]
    for length, token in enumerate(seq, 1):
        pools.out_tokens[:, length-1] = token
        pools.slen[:] = length
        store_ngram_keys(pools, idx, (4,))
        got = ngram_ban_pool(torch.zeros(2, 128), pools, idx, (4,)) < -1e8
        grams = Counter(tuple(seq[i:i+4]) for i in range(length-3))
        for row, limit in enumerate([1, 2]):
            expected = {g[-1] for g, count in grams.items() if g[:-1] == tuple(seq[:length][-3:]) and count >= limit}
            assert set(got[row].nonzero().flatten().tolist()) == expected
    # At the third occurrence both limits must ban 7; after only one, limit 2 allows it.
    assert got[:, 7].all()
    model = _model('dense')
    eng = _engine(model, slots=2, block=2)
    first = eng.submit_ids([4, 5], SamplingParams(n=1, max_ngram_occurrences=2, max_new_tokens=1))
    eng.run_until_complete()
    second = eng.submit_ids([4, 5], SamplingParams(n=1, max_new_tokens=1))
    eng.run_until_complete()
    assert eng.pools.ngram_max[second.s0] == 1
