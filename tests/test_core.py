"""CPU tests for the Phase-1 pipeline. No dataset download and no GPU required."""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.data.anonymize import anonymize_names_orgs, anonymize_statutes
from saul.data.loader import PackedLMDataset, collate_lm, cu_seqlens_np, mntp_mask
from saul.data.split import cut_dev_example
from saul.data.tokens import TokenSource
from saul.decode.mbr import _lcs, _lcs_dp, mbr_select, rouge_l
from saul.decode.postprocess import to_128_words
from saul.decode.processors import EtaSampling, GeneratedNoRepeatNGram, SuppressTokens, TopP, WindowedFrequencyPenalty
from saul.model.lm_head import next_token_targets
from saul.model.masks import bidirectional_document_mask, document_causal_mask, sliding_window_mask
from saul.model.modeling_saul import build_model, tiny_config
from saul.tokenizer.specials import build_special_tokens
from saul.tokenizer.train import GPT2_SINGLE_DIGIT, to_hf, train_bpe
from saul.train.optim import MUON_RMS, Muon, build_optimizers, zeropower_newtonschulz
from saul.train.schedule import wsd_multiplier
from saul.utils.params import count_params

ARCHS = ("dense", "hybrid", "ced", "conv")


# ------------------------------------------------------------------------------------------------ fixtures
def _unique_source(name: str, lengths: list[int], base: int = 100) -> TokenSource:
    """Every token value is unique, so an emitted token identifies (doc, offset)."""
    toks, index, cursor = [], [], 0
    for n in lengths:
        toks.append(np.arange(base + cursor, base + cursor + n))
        index.append((cursor, n, n, 0))
        cursor += n
    return TokenSource(name, np.concatenate(toks).astype(np.uint16), np.asarray(index, dtype=np.int64))


def _where(src: TokenSource, base: int = 100):
    starts = src.index[:, 0]

    def f(tok: int):
        j = int(np.searchsorted(starts, tok - base, side="right") - 1)
        return j, int(tok - base - starts[j])
    return f


@pytest.fixture(scope="module")
def tokenizer():
    specials = build_special_tokens(["<NAME>", "<ORG>"])
    corpus = [
        "Section 156(3) Cr.P.C. dated 19.8.2011. The <NAME> appeared for the appellant before the court.",
        "M/s. ABC Ltd. versus the State of Haryana (supra) Hon'ble Justice Rao dismissed the petition.",
        "AIR 1966 SC 1882. (a) The petitioner filed the present appeal against the order of the High Court.",
    ] * 50
    return to_hf(train_bpe(corpus, vocab_size=512, specials=specials, min_frequency=1, regex=GPT2_SINGLE_DIGIT)), corpus


# ------------------------------------------------------------------------------------------------ data
def test_dev_cut_word_counts():
    text = " ".join(f"w{i}" for i in range(2000))
    prompt, truth = cut_dev_example(text, "doc", 512, 128, 300, 781)
    p, t = prompt.split(), truth.split()
    assert len(p) == 512 and len(t) == 128
    assert int(t[0][1:]) == int(p[-1][1:]) + 1  # the truth continues exactly where the prompt stops


def test_anonymizer_keeps_institutions_and_masks_parties():
    text = "PETITIONER: Ashok Kumar\nHon'ble Justice Rao observed that Mr. Sharma appeared. State of Haryana vs. Ram Lal."
    out = anonymize_names_orgs(text)
    assert "Ashok" not in out and "<NAME>" in out and "State of Haryana" in out
    masked = anonymize_statutes("Section 302 of the Indian Penal Code, 1860 applies.", "<STATUTE>")
    assert "302" not in masked and "<STATUTE>" in masked


def test_packer_reads_long_documents_in_full_with_segment_positions():
    lengths = [100, 5, 7, 9, 6, 8, 40, 5]
    src = _unique_source("a", lengths)
    where = _where(src)
    ds = PackedLMDataset({"a": src}, {"a": 1}, seq_len=16, bos_id=0, eos_id=1, pad_id=2, seed=1,
                         min_remainder=1, worker_override=(0, 1))
    it = iter(ds)
    open_ends: dict[int, list[int]] = {j: [] for j in range(len(lengths))}
    seen: dict[int, set] = {j: set() for j in range(len(lengths))}
    for _ in range(80):
        s = next(it)
        assert (s["doc_ids"] >= 0).all(), "rows are always full"
        for d in np.unique(s["doc_ids"]):
            seg = s["input_ids"][s["doc_ids"] == d]
            pos = s["position_ids"][s["doc_ids"] == d]
            assert (pos == np.arange(len(seg))).all() and pos.max() < 16
            j, off = where(int(seg[0]))
            assert all(where(int(t)) == (j, off + i) for i, t in enumerate(seg)), "segment is a contiguous slice"
            if off > 0:
                assert off in open_ends[j], "a continuation resumes exactly where an earlier piece stopped"
                open_ends[j].remove(off)
            if off + len(seg) < lengths[j]:
                open_ends[j].append(off + len(seg))
            seen[j].update(range(off, off + len(seg)))
    assert seen[0] == set(range(100)), "the 100-token document was read in full"


def test_workers_read_disjoint_documents_and_resume_exactly():
    src = _unique_source("a", [3] * 40)
    where = _where(src)

    def starts(ds, n):
        it = iter(ds)
        out = set()
        for _ in range(n):
            s = next(it)
            for d in np.unique(s["doc_ids"]):
                j, off = where(int(s["input_ids"][s["doc_ids"] == d][0]))
                if off == 0:
                    out.add(j)
        return out

    mk = lambda w: PackedLMDataset({"a": src}, {"a": 1}, 8, 0, 1, 2, seed=3, worker_override=(w, 2))  # noqa: E731
    s0, s1 = starts(mk(0), 5), starts(mk(1), 5)
    assert s0 and s1 and not (s0 & s1)

    ds = mk(0)
    it = iter(ds)
    seqs = [next(it) for _ in range(7)]
    resumed = mk(0)
    w, W, st = seqs[3]["_wstate"]
    resumed.load_state_dict({"num_workers": W, "workers": {w: st}})
    it2 = iter(resumed)
    for k in range(4, 7):
        assert np.array_equal(next(it2)["input_ids"], seqs[k]["input_ids"])


def test_collate_and_cu_seqlens():
    src = _unique_source("a", [5, 9, 3, 12])
    ds = PackedLMDataset({"a": src}, {"a": 1}, 16, 0, 1, 2, seed=3, worker_override=(0, 1))
    it = iter(ds)
    col = collate_lm([next(it) for _ in range(2)], pad_id=2)
    assert col["input_ids"].shape == (2, 16) and col["labels"].shape == (2, 16)
    cu = col["cu_seqlens"].numpy()
    doc = col["doc_ids"].numpy()
    assert cu[0] == 0 and cu[-1] == 32 and np.all(np.diff(cu) > 0)
    for a, b in zip(cu[:-1], cu[1:]):
        flat = doc.reshape(-1)[a:b]
        assert len(set(flat.tolist())) == 1 and a // 16 == (b - 1) // 16
    assert np.array_equal(cu, cu_seqlens_np(doc))


def test_mntp_spans_stay_in_documents_and_hit_the_ratio():
    rng = np.random.default_rng(0)
    ids = torch.arange(10, 10 + 400).view(2, 200)
    docs = torch.zeros(2, 200, dtype=torch.int32)
    docs[:, 100:] = 1
    masked, labels, lm = mntp_mask(ids, docs, pad_id=2, mask_id=3, vocab_size=1000, rng=rng, random_low=64)
    frac = lm.float().mean().item()
    assert 0.25 <= frac <= 0.31
    assert torch.equal(labels, ids)
    assert ((masked != ids) <= lm).all(), "only masked positions change"
    assert ((masked == 3) & lm).float().sum() / lm.sum() > 0.7


def test_next_token_targets_alignment():
    labels = torch.tensor([[10, 11, 12, 13, 14, 15]])
    docs = torch.tensor([[0, 0, 0, 1, 1, 1]])
    tgt = next_token_targets(labels, docs)
    assert tgt.tolist() == [[11, 12, -100, 14, 15, -100]], "no target across the document boundary"
    loss_mask = torch.tensor([[False, True, False, False, True, False]])
    tgt = next_token_targets(labels, docs, loss_mask)
    # the masked inputs are at t=1 and t=4; they are predicted from t=0 and t=3
    assert tgt.tolist() == [[11, -100, -100, 14, -100, -100]]


# ------------------------------------------------------------------------------------------------ model
def test_masks_are_causal_inside_documents_only():
    doc = torch.tensor([[0, 0, 0, 1, 1, -1]])
    causal = document_causal_mask(doc)[0]
    assert bool(causal[2, 0]) and bool(causal[2, 2]) and not bool(causal[2, 3]) and not bool(causal[1, 2])
    bi = bidirectional_document_mask(doc)[0]
    assert bool(bi[0, 2]) and bool(bi[2, 0]) and not bool(bi[0, 3])
    win = sliding_window_mask(doc, window=2)[0]
    assert bool(win[2, 1]) and not bool(win[2, 0])


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_backward_and_budget(arch):
    cfg = tiny_config(arch)
    model = build_model(cfg, enforce_budget=True, enforce_min=False)
    B, T = 2, 16
    ids = torch.randint(4, cfg.vocab_size, (B, T))
    docs = torch.zeros(B, T, dtype=torch.int32)
    docs[:, 8:] = 1
    pos = torch.cat([torch.arange(8), torch.arange(8)]).view(1, -1).expand(B, -1)
    out = model(ids, labels=ids.clone(), doc_ids=docs, position_ids=pos)
    assert out.logits.shape == (B, T, cfg.vocab_size) and torch.isfinite(out.loss)
    out.loss.backward()
    assert count_params(model) < 299_000_000
    if arch == "dense":
        lm = torch.zeros(B, T, dtype=torch.bool)
        lm[:, 3] = True
        o2 = model(ids, labels=ids, doc_ids=docs, position_ids=pos, loss_mask=lm, mode="bidirectional")
        assert torch.isfinite(o2.loss)


@pytest.mark.parametrize("arch", ARCHS)
def test_documents_are_isolated(arch):
    """Changing document 0 must not change any logit of document 1 (attention, convs, GDN, Engram, residuals)."""
    torch.manual_seed(0)
    cfg = tiny_config(arch)
    model = build_model(cfg, enforce_min=False).eval()
    for p in model.parameters():  # zero-initialised output maps would hide leaks
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.05)
    T = 24
    ids = torch.randint(4, cfg.vocab_size, (1, T))
    docs = torch.zeros(1, T, dtype=torch.int32)
    docs[:, 10:] = 1
    pos = torch.cat([torch.arange(10), torch.arange(14)]).view(1, -1)
    base = model(ids, doc_ids=docs, position_ids=pos).logits
    ids2 = ids.clone()
    ids2[:, :10] = torch.randint(4, cfg.vocab_size, (1, 10))
    other = model(ids2, doc_ids=docs, position_ids=pos).logits
    assert torch.allclose(base[:, 10:], other[:, 10:], atol=1e-5)
    assert not torch.allclose(base[:, :10], other[:, :10], atol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_cached_decoding_matches_full_forward(arch):
    torch.manual_seed(0)
    cfg = tiny_config(arch)
    model = build_model(cfg, enforce_min=False).eval()
    for p in model.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.05)
    prompts = [torch.randint(4, cfg.vocab_size, (7,)).tolist(), torch.randint(4, cfg.vocab_size, (4,)).tolist()]
    forced = torch.randint(4, cfg.vocab_size, (2, 5))
    T = max(map(len, prompts))
    ids = torch.full((2, T), cfg.pad_token_id)
    valid = torch.zeros(2, T, dtype=torch.bool)
    for b, p in enumerate(prompts):
        ids[b, T - len(p):] = torch.tensor(p)
        valid[b, T - len(p):] = True
    logits, cache = model.prefill(ids, valid, max_new=6)
    steps = [logits]
    for j in range(4):
        steps.append(model.decode_step(forced[:, j], cache))
    for b, p in enumerate(prompts):
        full = model(torch.tensor([p + forced[b, :4].tolist()])).logits[0]
        for j in range(5):
            assert torch.allclose(steps[j][b], full[len(p) - 1 + j], atol=1e-4), (arch, b, j)
    # fanning one prefill out to several samples gives identical rows
    logits, cache = model.prefill(ids, valid, max_new=3)
    cache.expand(3)
    nxt = model.decode_step(forced[:, 0].repeat_interleave(3), cache)
    assert torch.allclose(nxt[0], nxt[2], atol=1e-5) and torch.allclose(nxt[3], nxt[5], atol=1e-5)


# ------------------------------------------------------------------------------------------------ optim
def test_newtonschulz_batched_matches_single_and_is_near_orthogonal():
    g = torch.randn(3, 8, 16)
    batched = zeropower_newtonschulz(g)
    single = torch.stack([zeropower_newtonschulz(x) for x in g])
    assert torch.allclose(batched, single, atol=2e-2)
    eye = torch.eye(8)
    gn = g[0] / g[0].norm()
    assert (batched[0] @ batched[0].T - eye).abs().mean() < (gn @ gn.T - eye).abs().mean()


def test_muon_update_rms_is_point_two_times_lr():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(128, 64) * 0.02)
    opt = Muon([p], lr=1.0, momentum=0.0, weight_decay=0.0)
    before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt.step()
    rms = (p.detach() - before).pow(2).mean().sqrt().item()
    assert 0.6 * MUON_RMS < rms < 1.4 * MUON_RMS, rms


def test_optimizer_groups_cover_everything_and_head_split_survives_reload():
    model = build_model(tiny_config("hybrid"), enforce_min=False)
    muon, adamw = build_optimizers(model, 2e-3, 2e-3)
    seen = {id(p) for g in muon.param_groups for p in g["params"]} | \
        {id(p) for g in adamw.param_groups for p in g["params"]}
    assert [n for n, p in model.named_parameters() if id(p) not in seen] == []
    q = model.blocks[3].mixer.q.weight
    assert muon.blocks[q] == model.config.n_heads
    x = torch.randint(4, 20, (2, 8))
    model(x, labels=x).loss.backward()
    muon.step()
    sd = muon.state_dict()
    muon2, _ = build_optimizers(model, 2e-3, 2e-3)
    muon2.load_state_dict(sd)
    assert muon2.blocks[q] == model.config.n_heads
    assert {g.get("name") for g in adamw.param_groups} >= {"embed", "norm", "other"}


def test_wsd_warmup_stable_decay():
    assert wsd_multiplier(0, 100) < wsd_multiplier(5, 100) <= 1
    assert abs(wsd_multiplier(50, 100) - 1) < 1e-6
    assert wsd_multiplier(99, 100) < 0.2


def test_save_load_roundtrip_is_exact(tmp_path):
    from saul.model.modeling_saul import SaulForCausalLM

    torch.manual_seed(0)
    model = build_model(tiny_config("ced"), enforce_min=False).eval()
    cmap = torch.arange(128)
    cmap[70:80] = 70
    model.engram.set_compression(cmap)
    model.save_pretrained(tmp_path / "m")
    back = SaulForCausalLM.from_pretrained(tmp_path / "m").eval()
    a, b = model.state_dict(), back.state_dict()
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])]
    assert back.lm_head.weight.data_ptr() == back.embed.weight.data_ptr()
    x = torch.randint(4, 128, (1, 12))
    assert torch.allclose(model(x).logits, back(x).logits, atol=1e-6)


def test_trainer_mntp_checkpoint_and_resume(tmp_path):
    from saul.data.loader import make_dataloader
    from saul.train.trainer import Trainer

    torch.manual_seed(0)
    src = _unique_source("a", [30, 12, 50, 9, 22, 17] * 3, base=4)
    src.tokens[:] = src.tokens % 120 + 4  # inside the tiny vocab
    cfg = {"seed": 1, "train": {"total_steps": 4, "grad_accum": 2, "seq_len": 32, "batch_size": 2, "log_every": 1,
                                "ckpt_every": 0, "eval_every": 0, "mntp": True, "mntp_every": 2, "device": "cpu"},
           "optim": {"muon_lr": 2e-3, "adamw_lr": 2e-3, "warmup_frac": 0.25, "decay_frac": 0.5}}
    ds = PackedLMDataset({"a": src}, {"a": 1}, 32, 0, 1, 2, seed=1)
    model = build_model(tiny_config("dense"), enforce_min=False)
    tr = Trainer(model, make_dataloader(ds, 2), cfg, str(tmp_path / "run"))
    stats = tr.train()
    assert stats["steps"] == 4 and tr.ema.updates == 2
    assert os.path.exists(tmp_path / "run" / "ckpt_final" / "trainer.pt")
    assert os.path.exists(tmp_path / "run" / "ckpt_stable")
    ds2 = PackedLMDataset({"a": src}, {"a": 1}, 32, 0, 1, 2, seed=1)
    tr2 = Trainer(build_model(tiny_config("dense"), enforce_min=False), make_dataloader(ds2, 2), cfg,
                  str(tmp_path / "run2"))
    tr2.load(str(tmp_path / "run" / "ckpt_stable"))
    assert tr2.step == 2 and tr2.data_state
    tr2.train()
    assert tr2.step == 4


# ------------------------------------------------------------------------------------------------ decoding
def test_ngram_block_is_vectorised_and_ignores_the_prompt():
    proc = GeneratedNoRepeatNGram(n=3)
    new = torch.tensor([[1, 2, 9, 1, 2]])
    out = proc(torch.zeros(1, 12), new)
    assert float(out[0, 9]) < -1e8 and (out[0, torch.arange(12) != 9] == 0).all()
    rng = random.Random(0)
    for _ in range(50):
        gen = [rng.randrange(6) for _ in range(rng.randrange(3, 30))]
        n = rng.choice([2, 3, 4])
        banned = {tuple(gen[i:i + n]) for i in range(len(gen) - n + 1)}
        expect = {g[-1] for g in banned if list(g[:-1]) == gen[len(gen) - n + 1:]} if len(gen) >= n else set()
        got = GeneratedNoRepeatNGram(n)(torch.zeros(1, 6), torch.tensor([gen]))
        assert {i for i in range(6) if got[0, i] < -1e8} == expect


def test_frequency_suppress_eta_topp():
    new = torch.tensor([[3, 3, 5], [1, 1, 1]])
    out = WindowedFrequencyPenalty(64, 0.5)(torch.zeros(2, 8), new)
    assert out[0, 3] == -1.0 and out[0, 5] == -0.5 and out[1, 1] == -1.5 and out[0, 0] == 0
    sup = SuppressTokens([0, 1, 2])(torch.zeros(2, 8), new)
    assert (sup[:, :3] < -1e8).all() and (sup[:, 3:] == 0).all()
    logits = torch.randn(4, 50) * 3
    for proc in (EtaSampling(3e-3), TopP(0.5)):
        kept = proc(logits, new) > -1e8
        assert kept[torch.arange(4), logits.argmax(-1)].all()


def test_bitparallel_lcs_matches_dp_and_mbr_picks_medoid():
    rng = random.Random(1)
    for _ in range(200):
        a = [rng.randrange(5) for _ in range(rng.randrange(0, 40))]
        b = [rng.randrange(5) for _ in range(rng.randrange(0, 40))]
        assert _lcs(a, b) == _lcs_dp(a, b)
    cands = ["a a a a", "a a a b", "zzz zzz zzz zzz"]
    assert mbr_select(cands, rouge_l).startswith("a a a")
    assert len(to_128_words(" ".join(f"w{i}" for i in range(200))).split()) == 128


def test_tokenizer_digits_are_atomic_and_roundtrip(tokenizer):
    tok, corpus = tokenizer
    for text in corpus[:3]:
        ids = tok.encode(text, add_special_tokens=False)
        assert tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text
    assert tok.convert_tokens_to_ids("<NAME>") == 11
    from saul.tokenizer.gates import gate_digits
    assert gate_digits(tok.backend_tokenizer, 64)["pass"]


def test_word_counter_generation_and_embedding_mbr(tokenizer):
    from saul.decode.generate import WordCounter, generate, suppressed_ids
    from saul.decode.mbr import embedding_matrix, mbr_choose
    from saul.decode.processors import build_logits_fn
    from saul.tokenizer.compress import build_compression_map

    tok, corpus = tokenizer
    V = 512
    counter = WordCounter(tok, V, "cpu")
    rng = np.random.default_rng(0)
    for _ in range(20):
        seq = torch.tensor(rng.integers(64, V, 30))
        words, prev = counter.init(1, "cpu")
        for t in seq:
            words, prev = counter.update(t.view(1), words, prev)
        text = tok.decode(seq.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert int(words) == len(text.split())
    cmap = build_compression_map(tok, V)
    assert cmap.shape == (V,) and (cmap[:64] == torch.arange(64)).all() and (cmap[cmap] == cmap).all()

    torch.manual_seed(0)
    model = build_model(tiny_config("ced", vocab_size=V), enforce_min=False).eval()
    model.engram.set_compression(cmap)
    fn = build_logits_fn({"temperature": 0.8, "truncation": "min_p"}, suppressed_ids(tok, V))
    outs = generate(model, tok, corpus[:2], n_samples=3, max_new_tokens=200, target_words=6, logits_fn=fn,
                    device="cpu", counter=counter)
    assert len(outs) == 2 and all(len(o) == 3 and all(len(t.split()) == 6 for t in o) for o in outs)
    assert not any(s in t for o in outs for t in o for s in ("<NAME>", "<ORG>"))
    U = embedding_matrix(model, tok, outs[0], "cpu")
    assert U.shape == (3, 3) and np.allclose(U, U.T, atol=1e-5)
    assert mbr_choose(outs[0], "mixed", 0.5, model, tok, "cpu") in outs[0]
