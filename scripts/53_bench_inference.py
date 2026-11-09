#!/usr/bin/env python
"""Verify, audit and benchmark the inference engine.

    python scripts/53_bench_inference.py --mode verify
    python scripts/53_bench_inference.py --mode coherence
    python scripts/53_bench_inference.py --mode bench
    python scripts/53_bench_inference.py --mode all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from saul.model.modeling_saul import SaulForCausalLM, build_model, tiny_config  # noqa: E402
from saul.serve.api import LLM  # noqa: E402
from saul.serve.sampling import MBRParams, SamplingParams  # noqa: E402
from saul.utils.io import read_json, write_json  # noqa: E402


def _load(checkpoint: str, tokenizer: str, **cfg):
    return LLM(checkpoint=checkpoint, tokenizer_path=tokenizer, **cfg)


def _prompts(path: str, n: int) -> list[str]:
    data = read_json(path)
    items = list(data.values()) if isinstance(data, dict) else data
    texts = []
    for item in items:
        texts.append(item["text"] if isinstance(item, dict) else item)
        if len(texts) >= n:
            break
    return texts


def verify_forward(checkpoint: str, tokenizer: str, prompts_path: str, out_dir: str, steps: int = 64,
                   n_prompts: int = 20) -> dict:
    """Graph vs eager vs full forward on one checkpoint. Teacher-forced so the three paths see the same tokens."""
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"
    model = SaulForCausalLM.from_pretrained(checkpoint).to(device).to(torch.bfloat16).eval()
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer)
    texts = _prompts(prompts_path, n_prompts)
    ids = []
    bos = tok.bos_token_id
    for t in texts:
        row = tok.encode(t, add_special_tokens=False)
        if bos is not None and (not row or row[0] != bos):
            row = [bos] + row
        ids.append(row[:1024])
    # greedy tokens from a full forward would be slow; force the argmax of a single eager engine, then replay
    eager = LLM(model, tok, max_slots=max(4, n_prompts * 2), block_size=2, max_prompt_len=1024, max_new_tokens=steps,
                prefill_token_budget=65536, cuda_graphs=False, device=device, dtype="bfloat16", rouge_workers=1)
    forced = []
    for row in ids:
        req = eager.engine.submit_ids(row, SamplingParams(
            n=2, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
            target_words=None, max_new_tokens=steps, record_logits=True))
        forced.append(req)
    eager.engine.run_until_complete()
    # Greedy with penalties off: the sampled id is the argmax of the recorded logits.
    sequences = [[int(row.argmax()) for row in req.traces[0]] for req in forced]
    del eager
    torch.cuda.empty_cache()

    def run(graphs: bool):
        llm = LLM(model, tok, max_slots=max(4, n_prompts * 2), block_size=2, max_prompt_len=1024,
                  max_new_tokens=steps, prefill_token_budget=65536, cuda_graphs=graphs, device=device,
                  dtype="bfloat16", rouge_workers=1)
        reqs = []
        for row, seq in zip(ids, sequences):
            reqs.append(llm.engine.submit_ids(row, SamplingParams(
                n=2, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
                target_words=None, max_new_tokens=len(seq), forced_tokens=tuple(seq), record_logits=True)))
        llm.engine.run_until_complete()
        traces = [r.traces for r in reqs]
        stats = llm.stats()
        llm.close()
        return traces, stats

    graph_traces, graph_stats = run(True)
    eager_traces, eager_stats = run(False)
    # full forward reference on the first slot of each prompt
    max_abs = 0.0
    mean_abs = 0.0
    argmax_ok = 0
    argmax_n = 0
    graph_eager_max = 0.0
    graph_eager_arg = 0
    count = 0
    for row, seq, gtr, etr in zip(ids, sequences, graph_traces, eager_traces):
        if not seq:
            continue
        full = model(torch.tensor([row + seq], device=device)).logits[0].float().cpu()
        for t in range(len(seq)):
            g = gtr[0][t].float()
            e = etr[0][t].float()
            ref = full[len(row) - 1 + t].cpu()
            max_abs = max(max_abs, (e - ref).abs().max().item())
            mean_abs += (e - ref).abs().mean().item()
            argmax_ok += int(e.argmax() == ref.argmax())
            argmax_n += 1
            graph_eager_max = max(graph_eager_max, (g - e).abs().max().item())
            graph_eager_arg += int(g.argmax() == e.argmax())
            count += 1
    report = {
        "checkpoint": checkpoint,
        "prompts": len(ids),
        "steps": steps,
        "eager_vs_forward_max_abs": max_abs,
        "eager_vs_forward_mean_abs": mean_abs / max(count, 1),
        "eager_vs_forward_argmax": argmax_ok / max(argmax_n, 1),
        "graph_vs_eager_max_abs": graph_eager_max,
        "graph_vs_eager_argmax": graph_eager_arg / max(count, 1),
        "graph_captures": graph_stats.get("graph_captures"),
        "pass_graph": graph_eager_arg == count and graph_eager_max <= 1e-2,
        "pass_forward": (argmax_ok / max(argmax_n, 1)) >= 0.99,
    }
    write_json(report, os.path.join(out_dir, "verify.json"))
    print(json.dumps(report, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()
    return report


def verify_tiny_graphs(out_dir: str) -> dict:
    """Random CED and conv models on GPU, graph vs eager argmax."""
    os.makedirs(out_dir, exist_ok=True)
    from tests.test_serve import Tok
    results = {}
    for arch in ("ced", "conv"):
        torch.manual_seed(0)
        model = build_model(tiny_config(arch), enforce_min=False).cuda().to(torch.bfloat16).eval()
        for p in model.parameters():
            if p.ndim >= 2:
                torch.nn.init.normal_(p, std=0.05)
        tok = Tok()
        prompt = [4, 5, 6, 7, 8, 9]
        forced = tuple(range(10, 26))

        def once(graphs):
            llm = LLM(model, tok, max_slots=4, block_size=2, max_prompt_len=32, max_new_tokens=16,
                      prefill_token_budget=128, cuda_graphs=graphs, device="cuda", dtype="bfloat16", rouge_workers=1)
            req = llm.engine.submit_ids(prompt, SamplingParams(
                n=2, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
                target_words=None, max_new_tokens=len(forced), forced_tokens=forced, record_logits=True))
            llm.engine.run_until_complete()
            trace = req.traces
            llm.close()
            return trace

        g, e = once(True), once(False)
        agree = sum(int(torch.equal(a.argmax(-1), b.argmax(-1))) for a, b in zip(g[0], e[0]))
        results[arch] = {"steps": len(g[0]), "argmax_agree": agree, "pass": agree == len(g[0])}
        del model
        torch.cuda.empty_cache()
    write_json(results, os.path.join(out_dir, "verify_tiny.json"))
    print(json.dumps(results, indent=2), flush=True)
    return results


def old_greedy(model, tok, prompts: list[str], max_new: int = 64) -> list[list[int]]:
    """The previous cache loop, greedy, no repetition penalties. Used only as a numeric baseline."""
    device = next(model.parameters()).device
    bos, pad = tok.bos_token_id, tok.pad_token_id
    enc = []
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        enc.append(ids)
    T = max(len(x) for x in enc)
    batch = torch.full((len(enc), T), pad, dtype=torch.long, device=device)
    valid = torch.zeros((len(enc), T), dtype=torch.bool, device=device)
    for i, x in enumerate(enc):
        batch[i, T - len(x):] = torch.tensor(x, device=device)
        valid[i, T - len(x):] = True
    logits, cache = model.prefill(batch, valid, max_new)
    out = [[] for _ in enc]
    for _ in range(max_new):
        nxt = logits.float().argmax(-1)
        for i, tok_id in enumerate(nxt.tolist()):
            out[i].append(tok_id)
        logits = model.decode_step(nxt, cache)
    return out


def verify_old(checkpoint: str, tokenizer: str, prompts_path: str, out_dir: str, n: int = 50) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    model = SaulForCausalLM.from_pretrained(checkpoint).cuda().to(torch.bfloat16).eval()
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer)
    texts = _prompts(prompts_path, n)
    old = old_greedy(model, tok, texts, max_new=48)
    llm = LLM(model, tok, max_slots=n, block_size=1, max_prompt_len=1024, max_new_tokens=48,
              cuda_graphs=True, device="cuda", dtype="bfloat16", rouge_workers=1)
    outs = llm.generate(texts, SamplingParams(
        n=1, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
        target_words=None, max_new_tokens=48), prepend_bos=True)
    prefixes, full = [], 0
    for old_ids, out in zip(old, outs):
        new_ids = tok.encode(out.candidates[0], add_special_tokens=False)[:48]
        k = 0
        for a, b in zip(old_ids, new_ids):
            if a != b:
                break
            k += 1
        prefixes.append(k)
        full += int(old_ids == new_ids)
    report = {"n": n, "mean_matching_prefix": sum(prefixes) / max(len(prefixes), 1),
              "full_agree": full / max(n, 1), "min_prefix": min(prefixes) if prefixes else 0}
    write_json(report, os.path.join(out_dir, "verify_old.json"))
    print(json.dumps(report, indent=2), flush=True)
    llm.close()
    del model
    torch.cuda.empty_cache()
    return report


def coherence(checkpoint: str, tokenizer: str, prompts_path: str, truth_path: str | None, out_dir: str,
              n: int = 500) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    texts = _prompts(prompts_path, n)
    llm = _load(checkpoint, tokenizer, max_slots=128, block_size=1, max_prompt_len=2048, max_new_tokens=1024,
                device="cuda", dtype="bfloat16", cuda_graphs=True, rouge_workers=1)
    kept, skipped = [], 0
    for text in texts:
        try:
            llm.engine.tokenize([text])
        except ValueError:
            skipped += 1
            continue
        kept.append(text)
    texts = kept
    t0 = time.perf_counter()
    outs = llm.generate(texts, SamplingParams(n=1, temperature=0.0, truncation="none", target_words=128,
                                              max_new_tokens=1024, no_repeat_ngram=4, frequency_penalty=0.3))
    dt = time.perf_counter() - t0
    specials = ("<|", "<NAME>", "<ORG>", "<SECTION>", "<ENTITY>", "<ACT>", "<CITATION>")
    words = [o.text.split() for o in outs]
    exact = sum(len(w) == 128 for w in words)
    leaks = sum(any(s in o.text for s in specials) for o in outs)
    # repetition
    rep8 = 0
    distinct2 = set()
    for w in words:
        grams = [tuple(w[i:i + 8]) for i in range(len(w) - 7)]
        if len(grams) != len(set(grams)):
            rep8 += 1
        distinct2.update(tuple(w[i:i + 2]) for i in range(len(w) - 1))
    report = {"n": len(outs), "skipped_long": skipped, "seconds": round(dt, 2), "exact_128": exact, "special_leaks": leaks,
              "frac_repeated_8gram": rep8 / max(len(words), 1), "distinct_2_types": len(distinct2),
              "finish": {}}
    for o in outs:
        for r in o.finish_reasons:
            report["finish"][r] = report["finish"].get(r, 0) + 1
    if truth_path and os.path.exists(truth_path):
        from saul.eval.official import phase1_score
        pred = {str(i): {"text": o.text} for i, o in enumerate(outs)}
        # truth ids may not be 0..n-1; score only when the files share ids
        write_json({str(i + 1): {"text": o.text} for i, o in enumerate(outs)}, os.path.join(out_dir, "coherence_pred.json"))
        report["note"] = "prediction written; official score needs matching ids"
        del pred
    # ten samples for the hand review
    samples = []
    for text, out in list(zip(texts, outs))[:10]:
        tail = " ".join(text.split()[-24:])
        samples.append({"prompt_tail": tail, "continuation": " ".join(out.text.split()[:40])})
    report["samples"] = samples
    write_json(report, os.path.join(out_dir, "coherence.json"))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2), flush=True)
    llm.close()
    return report


def microbench(checkpoint: str, tokenizer: str, out_dir: str) -> dict:
    llm = _load(checkpoint, tokenizer, max_slots=512, block_size=16, max_prompt_len=1024, max_new_tokens=1024,
                device="cuda", dtype="bfloat16", cuda_graphs=True, rouge_workers=1)
    rows = []
    engine = llm.engine
    for sb in (16, 64, 128, 256, 512):
        for ls in (128, 512, 1024):
            lp = 1024
            p = engine.pools
            p.needs_decode[:sb] = True
            p.slen[:sb] = ls // 2
            groups = sb // p.B
            p.plen[:groups] = min(lp, 1024)
            p.plen_slot[:sb] = p.plen[(torch.arange(sb, device=p.plen.device) // p.B)]
            p.pos[:sb] = p.plen_slot[:sb] + p.slen[:sb] - 1
            p.token_ids[:sb] = 4
            bucket_lp = engine.cfg.bucket(lp, engine.cfg.prompt_buckets)
            bucket_ls = engine.cfg.bucket(ls // 2, engine.cfg.suffix_buckets)
            bucket_sb = engine.cfg.bucket(sb, engine.cfg.slot_buckets)
            for _ in range(3):
                engine.runner.decode(bucket_sb, bucket_lp, bucket_ls)
            torch.cuda.synchronize()
            steps = 10
            t0 = time.perf_counter()
            for _ in range(steps):
                engine.runner.decode(bucket_sb, bucket_lp, bucket_ls)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / steps * 1000
            rows.append({"slots": bucket_sb, "prompt": bucket_lp, "suffix": bucket_ls, "ms": round(ms, 3),
                         "tok_s": round(1000 / ms * bucket_sb, 1)})
            print(rows[-1], flush=True)
            p.needs_decode[:sb] = False
    eager_ms = None
    engine.runner.use_graphs = False
    p = engine.pools
    p.needs_decode[:512] = True
    p.slen[:512] = 512
    p.pos[:512] = 512
    p.token_ids[:512] = 4
    for _ in range(2):
        engine.runner.decode(512, 1024, 1024)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        engine.runner.decode(512, 1024, 1024)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / 5 * 1000
    report = {"graph": rows, "eager_512_1024_ms": round(eager_ms, 3)}
    write_json(report, os.path.join(out_dir, "microbench.json"))
    llm.close()
    return report


def profile_decode(checkpoint: str, tokenizer: str, out_dir: str) -> dict:
    llm = _load(checkpoint, tokenizer, max_slots=128, block_size=16, max_prompt_len=1024, max_new_tokens=256,
                device="cuda", dtype="bfloat16", cuda_graphs=True, rouge_workers=1)
    eng = llm.engine
    p = eng.pools
    sb, lp, ls = 128, 1024, 128
    p.needs_decode[:sb] = True
    p.slen[:sb] = 32
    p.plen[:sb // p.B] = 512
    p.plen_slot[:sb] = p.plen[(torch.arange(sb, device=p.device) // p.B)]
    p.pos[:sb] = p.plen_slot[:sb] + 31
    p.token_ids[:sb] = 4
    bsb = eng.cfg.bucket(sb, eng.cfg.slot_buckets)
    blp = eng.cfg.bucket(512, eng.cfg.prompt_buckets)
    bls = eng.cfg.bucket(32, eng.cfg.suffix_buckets)
    eng.runner.decode(bsb, blp, bls)  # capture
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(20):
            eng.runner.decode(bsb, blp, bls)
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25)
    launches = sum(1 for e in prof.key_averages() if "cudaGraphLaunch" in e.key)
    text = table
    path = os.path.join(out_dir, "profile.txt")
    open(path, "w").write(text)
    llm.close()
    return {"graph_launch_rows": launches, "profile": path}


def end_to_end(checkpoint: str, tokenizer: str, test_path: str, out_dir: str, limit: int = 0) -> dict:
    texts = _prompts(test_path, 10**9)
    if limit:
        texts = texts[:limit]
    results = {}
    specs = [
        ("n1_minp", SamplingParams(n=1, temperature=0.8, truncation="min_p", target_words=128, max_new_tokens=1024), 1),
        ("n1_greedy", SamplingParams(n=1, temperature=0.0, truncation="none", target_words=128, max_new_tokens=1024), 1),
        ("n16_mbr", SamplingParams(n=16, temperature=0.9, truncation="min_p", target_words=128, max_new_tokens=1024,
                                   mbr=MBRParams(enabled=True)), 16),
    ]
    for name, params, n in specs:
        block = 16 if n > 1 else 1
        slots = 512
        llm = _load(checkpoint, tokenizer, max_slots=slots, block_size=block, max_prompt_len=2048,
                    max_new_tokens=1024, device="cuda", dtype="bfloat16", cuda_graphs=True, rouge_workers=6)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        outs = llm.generate(texts, params)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = sum(sum(o.gen_tokens) for o in outs)
        results[name] = {"seconds": round(dt, 2), "prompts": len(texts), "generated_tokens": gen,
                         "tok_s": round(gen / dt, 1), "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                         "stats": llm.stats()}
        print(name, results[name]["seconds"], "s", results[name]["tok_s"], "tok/s", flush=True)
        llm.close()
        torch.cuda.empty_cache()
    write_json(results, os.path.join(out_dir, "e2e.json"))
    return results


def write_reports(out_dir: str) -> None:
    verify = json.load(open(os.path.join(out_dir, "verify.json"))) if os.path.exists(os.path.join(out_dir, "verify.json")) else {}
    micro = json.load(open(os.path.join(out_dir, "microbench.json"))) if os.path.exists(os.path.join(out_dir, "microbench.json")) else {}
    e2e = json.load(open(os.path.join(out_dir, "e2e.json"))) if os.path.exists(os.path.join(out_dir, "e2e.json")) else {}
    coh = json.load(open(os.path.join(out_dir, "coherence.json"))) if os.path.exists(os.path.join(out_dir, "coherence.json")) else {}
    lines = ["# Inference benchmark", ""]
    if e2e:
        lines.append("## End to end")
        lines.append("")
        lines.append("| job | seconds | tok/s | peak MiB |")
        lines.append("|---|---:|---:|---:|")
        for k, v in e2e.items():
            lines.append(f"| {k} | {v['seconds']} | {v['tok_s']} | {v['peak_mib']} |")
        lines.append("")
        lines.append("The previous MBR-16 submission (S2) took 2431 s on the old engine.")
        lines.append("")
    if micro:
        lines.append("## Decode step")
        lines.append("")
        lines.append("| slots | suffix bucket | ms | tok/s |")
        lines.append("|---:|---:|---:|---:|")
        for row in micro.get("graph", []):
            lines.append(f"| {row['slots']} | {row['suffix']} | {row['ms']} | {row['tok_s']} |")
        lines.append("")
        lines.append(f"Eager S=512 Lp=1024 Ls=1024: {micro.get('eager_512_1024_ms')} ms.")
        lines.append("")
    if verify:
        lines += ["## Parity", "", "```", json.dumps(verify, indent=2), "```", ""]
    if coh:
        lines += ["## Coherence", "", f"{coh.get('exact_128')}/{coh.get('n')} outputs are 128 words, "
                  f"{coh.get('special_leaks')} special-token leaks, "
                  f"repeated-8gram fraction {coh.get('frac_repeated_8gram')}.", ""]
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "benchmark.md"), "w").write("\n".join(lines) + "\n")
    audit = ["# Inference engine audit", "",
             "CPU tests cover logit parity for dense, hybrid, CED+Engram and conv, slot reuse, sampler equivalence, stopping, MBR and the HTTP API.",
             "",
             "Graph rules: `serve_decode` does not call `.item()` or branch on tensor values. Bucket choice and the done-flag copy are the host syncs outside the graph.",
             "",
             "Numerics: cascade attention softmax is fp32, GDN state is fp32, sampling logits are fp32.",
             ""]
    if verify:
        audit.append(f"GPU eager vs full forward argmax agreement: {verify.get('eager_vs_forward_argmax')}.")
        audit.append(f"GPU graph vs eager argmax agreement: {verify.get('graph_vs_eager_argmax')}, max abs {verify.get('graph_vs_eager_max_abs')}.")
    prof = os.path.join(out_dir, "profile.txt")
    if os.path.exists(prof):
        audit += ["", "## Profiler (20 replayed decode steps)", "", "```", open(prof).read()[:4000], "```"]
    open(os.path.join(out_dir, "audit.md"), "w").write("\n".join(audit) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["verify", "coherence", "bench", "micro", "profile", "after_verify", "all"])
    ap.add_argument("--checkpoint", default="runs/dense_286m/anneal_v2/ckpt_final_ema")
    ap.add_argument("--tokenizer", default="artifacts/tokenizers/legal32k")
    ap.add_argument("--prompts", default="data/final/dev_prompts_fast.json")
    ap.add_argument("--test", default="data/raw/Phase_1_Pretraining/Test/test.json")
    ap.add_argument("--out", default="artifacts/inference")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode in ("verify", "all"):
        verify_forward(a.checkpoint, a.tokenizer, a.prompts, a.out)
        verify_tiny_graphs(a.out)
        verify_old(a.checkpoint, a.tokenizer, a.prompts, a.out, n=20 if a.limit else 50)
    if a.mode in ("coherence", "after_verify", "all"):
        coherence(a.checkpoint, a.tokenizer, a.prompts, None, a.out, n=a.limit or 500)
    if a.mode in ("micro", "bench", "after_verify", "all"):
        microbench(a.checkpoint, a.tokenizer, a.out)
    if a.mode in ("profile", "after_verify", "all"):
        profile_decode(a.checkpoint, a.tokenizer, a.out)
    if a.mode in ("bench", "after_verify", "all"):
        end_to_end(a.checkpoint, a.tokenizer, a.test, a.out, limit=a.limit)
    write_reports(a.out)


if __name__ == "__main__":
    main()
