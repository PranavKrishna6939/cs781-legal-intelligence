# In-House Legal Intelligence System (CS781 Capstone)

Pranav Krishna (230772), IIT Kanpur.

This repository holds the code for all seven phases of the CS781 capstone. In Phase 1 we pretrain small
language models from scratch on Indian legal text. Phases 2 to 7 reuse those backbones for six downstream
legal tasks. No outside pretrained model, LLM or API is used anywhere; every weight comes from our own
pretraining.

| Phase | Task | Best submission (code in this repo) | Primary metric |
| --- | --- | --- | --- |
| 1 | Legal language model pretraining (masked span infilling) | `B_anneal_ema_mbr64_20261005` | BERTScore F1 0.8518 |
| 2 | Legal named entity recognition | `P2_E2_dense_lorma` | Strict macro F1 0.6749 |
| 3 | Rhetorical role labelling | `P3_E1_dense_lorma_dissent` | Macro F1 0.6657 |
| 4 | Court judgment prediction | `P4_dense_views_noleak` | Macro F1 0.8517 |
| 5 | Legal statute identification | `P5_E2_zlex` | Macro F1 0.4746 |
| 6 | Judgment summarisation | `H1_mbr32` | BERTScore F1 0.8534 |
| 7 | Prior case retrieval | `P7_R5_reporter_citator` | F1@7 0.4944 |

Model weights, checkpoints, the competition data and run outputs are not in this repository.
[`SUBMISSIONS.md`](SUBMISSIONS.md) lists every submission we made in every phase, with its score where we
recorded one.

## Repository layout

```
saul/                    library ("Saul" is the model family)
  model/                 Dense, Hybrid, CED and Conv-Mini architectures, Gated Residual, factory
  data/ tokenizer/       corpus cleaning, dedup, anonymisation, splits, BPE tokenizer, token packing
  train/                 pretraining loop: Muon + AdamW, WSD schedule, EMA, checkpointing
  serve/ decode/         batched generation engine, sampling, MBR decoding
  eval/ utils/           metrics, config loading, IO
  ner/                   Phase 2: LoRMA+ and other PEFT wrappers, CRF + biaffine heads, decoding
  rr/                    Phase 3: sentence role tagger, document decoder, dissent rule
  cjpe/                  Phase 4: long document classifier, views, pooling heads, mixing
  lsi/                   Phase 5: multi label statute classifier, retrieval, thresholds
  summ/                  Phase 6: summariser training and MBR decoding
configs/                 data mixtures, tokenizer, model and training configs
scripts/                 numbered pipeline scripts (Phase 1) and pN_* scripts (Phases 2 to 7)
tests/                   unit tests
resources/evaluation_script/   official course evaluation scripts
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install fastapi uvicorn lightgbm scipy flash-linear-attention
```

Python 3.11, PyTorch 2.x with CUDA. Most scripts assume the repository lives at `~/cs781`; set
`SAUL_ROOT` to override. Large outputs go to per phase directories that can be overridden with
`SAUL_CKPT_ROOT`, `SAUL_TOKENIZER`, `SAUL_P2_DIR`, `SAUL_P3_DIR`, `SAUL_P4_DIR`, `SAUL_P5_DIR`,
`SAUL_P6_DIR` and `SAUL_P6_DATA`. Competition data goes under `data/raw/Phase_N_*`.

Hardware: pretraining and large runs used one NVIDIA RTX PRO 6000 Blackwell (96 GB; three of them
until 5 October); a 24 GB A10G ran the PEFT runs in Phases 5 and 6. Phase 7 is CPU only.

<!-- phase1 -->
## Phase 1: legal language model pretraining

**Task.** Fill masked spans in Indian legal documents. Scored with BERTScore F1 (primary), ROUGE-L and BLEU-4.

**Data pipeline.** `scripts/00_download.py` → `01_clean.py` (Unicode and layout repair) →
`02_dedup.py` (exact and MinHash near duplicate removal) → `03_anonymize.py` (names and identifiers
replaced with typed tags that match the downstream data) → `04_split.py` → `05_stats.py`. A 32,768 token
byte level BPE tokenizer (`legal32k`) is trained with `10_train_tokenizer.py`, checked with
`10b_tokenizer_gates.py`, and the corpus is packed into 2,048 token sequences with `11_tokenize_corpus.py`.

**Architectures** (`saul/model`, configs in `configs/models`). Four families were pretrained:

* *Dense* (286M): 20 layer pre norm decoder, grouped query attention, SwiGLU, RoPE.
* *Hybrid* (294M, the best model): 20 layers in the pattern `[GDN, GDN, GDN, attention] x 5`, width 1024.
  Gated DeltaNet layers have 8 heads of size 128. Attention layers use GQA with 16 query and 4 key value
  heads (head size 64), partial RoPE on 32 of 64 dimensions and a sigmoid output gate. SwiGLU width 2304.
  A Gated Residual stream (n = 4, rank 64) replaces plain residual addition. Vocabulary 32,768 with tied
  embeddings; 294,335,856 parameters.
* *CED* (253M): compressed encoder decoder variant.
* *Conv-Mini* (99M): small convolution mixing model.

**Training** (`saul/train`, `scripts/30_train.py`). Muon (lr 2e-3) for matrices and AdamW for the rest, a
warmup stable decay schedule, 524,288 tokens per step (2,048 x 8 x 32), and an EMA of the weights (0.999)
kept only during the decay. The best checkpoint was trained in three stages:

1. Stable phase on the base mixture, steps 0 to 7,200:
   `python scripts/30_train.py --config configs/train/hybrid.yaml run_dir=runs/hybrid_294m/stable train.mixture=stable --stop_at_stable`
2. Continuation on a Supreme Court heavier mixture (`configs/data_stable_sc_heavy.yaml`) to step 19,202:
   `bash scripts/71_cont_b.sh`
3. Anneal on a mixture matched to the test distribution (`configs/data_anneal_testmix.yaml`), steps 19,202
   to 21,002, writing `ckpt_final_ema`: `bash scripts/95_recovery_b_anneal.sh`

**Decoding** (`saul/serve`, `saul/decode`). `scripts/109_make_recovery_submission.py --checkpoint <ckpt> --out <dir>`
starts the batched server (`52_serve.py`, prompts up to 2,048 tokens, left truncation) and runs
`91_recovery_decode.py`: 64 samples per span at temperature 0.9 with min p 0.05, no repeated 4-grams and a
frequency penalty of 0.3 over a 64 token window, then minimum Bayes risk selection with a mixed utility
(alpha 0.5), seed 781. The output is checked with `51_make_submission.py` before zipping.

**Result.** `B_anneal_ema_mbr64_20261005`: BERTScore 0.8518, ROUGE-L 0.2477, BLEU-4 0.1296.
<!-- /phase1 -->

<!-- phase2 -->
## Phase 2: legal named entity recognition

**Task.** Tag legal entities (judge, court, petitioner, statute, provision, precedent and so on) in
judgments. Scored with strict span macro F1.

**Model** (`saul/ner`). The Dense Phase 1 backbone run bidirectionally (causal mask removed), adapted with
**LoRMA+** (`saul/ner/peft.py`, `wrap_lorma`): a multiplicative low rank update
h = W0 x + (alpha / r) B A (W0 x) with rank 16 and alpha 32 on the q, k, v, gate, output and MLP
projections; embeddings, norms and Gated Residual linears are also trained (about 62M trainable
parameters). The head is a BIOES CRF over 49 tags with an auxiliary biaffine span loss.

**Post processing.** A span exact token decoder, a header parser for party and court fields, and a rule
based role resolver (`scripts/p2_post.py`).

**Commands.**
```bash
python -m saul.ner.train --exp E2v2 --model dense_v1 --technique lorma --context bidirectional --fold all --batch-size 8 --accum 2 --epochs 10
python scripts/p2_post.py E2v2 --save E2v2 --nested
python -m saul.ner.submit --exp E2v2 --name P2_E2_dense_lorma
```

**Result.** `P2_E2_dense_lorma`: strict macro F1 0.6749 (P 0.6925, R 0.6746); nested out of fold 0.626.
<!-- /phase2 -->

<!-- phase3 -->
## Phase 3: rhetorical role labelling

**Task.** Assign each sentence of a judgment a rhetorical role (facts, arguments, ratio, ruling and so
on). Scored with macro F1.

**Model** (`saul/rr`). The Dense backbone, bidirectional, with LoRMA+ (rank 16, alpha 32). Sentences are
pooled from token states and a document level decoder (`saul/rr/decode.py`) smooths the role sequence.

**Dissent rule.** In Competition Commission (CCI) orders, a dissenting note usually follows the phrase
"secretary is directed". When at least 10 sentences follow it, they are relabelled as dissent; see
`saul/rr/decode.py`. This raised out of fold macro F1 from
0.550 to 0.634.

**Commands.**
```bash
python -m saul.rr.train --exp E1 --model dense_v1 --technique lorma --context bidirectional --fold all --epochs 10 --g1-min 0.415
python scripts/p3_batch_zips.py --dissent
```
Training was driven by `scripts/p3_queue.py` (batch 8, accumulation 2). Fold test caches are merged with
`saul.rr.infer.merge_test`.

**Result.** `P3_E1_dense_lorma_dissent`: macro F1 0.6657 (0.5833 without the rule).
<!-- /phase3 -->

<!-- phase4 -->
## Phase 4: court judgment prediction

**Task.** Predict whether an appeal is accepted from the judgment text. Scored with macro F1.

**Model** (`saul/cjpe`). The Dense backbone, bidirectional, with LoRMA+. A pooling head (attention pooling
plus mean pooling) reads the tail of the judgment; the loss is logit adjusted cross entropy with class
prior 0.59 / 0.41. Dev cases that also appear in the test set were removed before model selection
("noleak", selection score D-sel).

**Commands.**
```bash
python -m saul.cjpe.data --prepare --workers 8
python -m saul.cjpe.train --exp C1 --model dense_v1 --technique lorma --context bidirectional --view tail --limit 2048 --epochs 2 --no-ovl --batch-tokens 16384 --accum 4 --compile --attn sdpa --gate 0.74 --first-limit 1024
python -m saul.cjpe.infer --source C1 --name C1_t4096 --limit 4096 --batch-tokens 16384
python scripts/p4_views_submit.py
```
The last step averages the log odds of the same model read at a 2,048 token and a 4,096 token tail.

**Result.** `P4_dense_views_noleak`: macro F1 0.8517 (P 0.8585, R 0.8438); D-sel 0.8540.
<!-- /phase4 -->

<!-- phase5 -->
## Phase 5: legal statute identification

**Task.** Multi label prediction of the statutes cited by a case. Scored with macro F1.

**Model** (`saul/lsi`). The CED backbone fully fine tuned with a decoder head over document segments
(512 tokens, up to 4 chunks in training and 8 at inference), BCE loss, and a k nearest neighbour retrieval
term (k = 8) over training states. Its logits are blended with cross fitted lexical margins from a word
1 to 2-gram TF-IDF one vs rest LinearSVC (C = 0.5, `scripts/p5_side.py`) with weight lambda = 0.45.
Per label thresholds are fitted on each half of dev, shrunk toward the global threshold, averaged, and
every case gets at least one label.

**Commands.**
```bash
python -m saul.lsi.data --prepare --workers 4
python scripts/p5_side.py
python -m saul.lsi.train --exp E2_ced --model ced --technique full --head decoder --view segments --limit 512 --compile-blocks --batch-tokens 16384 --accum 8 --epochs 3 --eval-per-epoch --no-wandb --max-chunks 4 --infer-chunks 8 --head-lr 5e-4 --embed-lr 1e-3 --adapter-lr 5e-4 --base-lr 4e-5 --lr-floor 0.1 --decay 0.85 --attn sdpa --ckpt-every 0 --loss bce --retrieval --knn 8 --train-logits --train-states
python scripts/p5_e2_zlex_submit.py
```

**Result.** `P5_E2_zlex`: macro F1 0.4746, micro F1 0.5710 (rank 1); cross fitted dev 0.4715.
<!-- /phase5 -->

<!-- phase6 -->
## Phase 6: judgment summarisation

**Task.** Write an abstractive summary of a judgment. Scored with BERTScore F1 (primary) and ROUGE.

**Model** (`saul/summ`). The Hybrid backbone with LoRMA+ (rank 16, alpha 32, only A and B trained,
about 6.6M parameters). The input is the first 2K and last 2K tokens of the judgment, the loss covers only
the summary, AdamW at lr 5e-4, and checkpoints are chosen on a 200 document holdout.

**Commands.**
```bash
P6_BUDGET_MIN=40 python -m saul.summ.run
H1_N=32 H1_SLOTS=64 python -m saul.summ.h1_mbr
```
Decoding samples 32 candidates (temperature 0.7, min p 0.05, no repeated 4-grams, frequency penalty 0.3
over 64 tokens, 1,200 words) and picks one by MBR with a mixed utility (alpha 0.6).

**Result.** `H1_mbr32`: BERTScore 0.8534, ROUGE-1 0.4837, ROUGE-L 0.2407.
<!-- /phase6 -->

<!-- phase7 -->
## Phase 7: prior case retrieval

**Task.** For each query judgment, return the prior cases it cites from a candidate pool. Scored with F1@7.

**Method.** A lexical learning to rank system, CPU only. Candidates come from Okapi BM25 over 3 to 5-gram
views of full text, opening sentences and citation contexts (`scripts/p7_bm25.py`, `p7_r1.py`). Each
query candidate pair gets 39 features (per query z scored BM25 views, rank features, hubness statistics,
train co citation, citator edges). A LightGBM lambdarank model is trained on train queries and bagged
over 5 seeds (`p7_r2.py`). Reporter citation links found in the query text are forced into the top 7
(`p7_r5.py`).

**Commands.**
```bash
python scripts/p7_e0.py
python scripts/p7_r1.py
P7_SHORT=200 P7_TAG=_s200 python scripts/p7_r2.py
python scripts/p7_probe8.py
P7_FORCE_ZIP=1 python scripts/p7_r5.py
```

**Result.** `P7_R5_reporter_citator`: F1@7 0.4944; dev 0.484.
<!-- /phase7 -->

## Report

The full write up is in the project report (`CS781-CP-Pranav-Krishna-230772-report.pdf`).
Weights are not distributed; every phase can be retrained from the commands above.
