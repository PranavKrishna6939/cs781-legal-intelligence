# All submissions

Scores are leaderboard values as recorded (primary metric first). A dash means the score was not recorded.
The submission whose code is described in the README is marked **best**.

## Phase 1 (BERTScore F1 ; ROUGE-L ; BLEU-4)

Names: `S`/`D` Dense, `H` Hybrid, `C` Conv-Mini, `CED` compressed encoder decoder, `A`/`B` continuation
runs, `soup` weight average. Suffixes give the decoder: `greedy`, `minp` sampling, `mbrN` MBR over N samples.

| Id | Submission | Scores |
| --- | --- | --- |
| 951256 | S2_dense_v2_ema_mbr16 | 0.8485 ; 0.2288 ; 0.1057 |
| 951395 | S4_dense_v2ema_greedy | 0.8429 ; 0.2208 ; 0.1016 |
| 951435 | S1_dense_v2ema_minp | 0.8426 ; 0.2112 ; 0.0905 |
| 951436 | S3_dense_v1ema_mbr16 | 0.8493 ; 0.2319 ; 0.1102 |
| 951749 | H1_hybrid_extendema_minp | 0.8430 ; 0.2134 ; 0.0921 |
| 951783 | H2_hybrid_extendema_mbr16 | 0.8493 ; 0.2320 ; 0.1099 |
| 951785 | H3_hybrid_annealema_mbr16 | 0.8495 ; 0.2350 ; 0.1174 |
| 951787 | H4_hybrid_extendema_greedy | 0.8441 ; 0.2228 ; 0.1028 |
| 953661 | C1_conv_minp | 0.8363 ; 0.1883 ; 0.0550 |
| 953662 | C2_conv_mbr16 | 0.8427 ; 0.2020 ; 0.0664 |
| 953663 | C3_conv_greedy | 0.8344 ; 0.1955 ; 0.0660 |
| 955466 | CED2_ema_mbr16 | 0.8490 ; 0.2303 ; 0.1089 |
| 955755 | A_conta_stable_mbr16 | 0.8465 ; 0.2215 ; 0.0943 |
| 955757 | B_conta_stable_mbr16 | 0.8467 ; 0.2224 ; 0.0935 |
| 955796 | H7_hybrid_sc55_mbr16 | 0.8493 ; 0.2350 ; 0.1158 |
| 955800 | H6_hybrid_polish_mbr16 | 0.8477 ; 0.2293 ; 0.1061 |
| 955803 | D5_dense_clm_mbr16 | 0.8491 ; 0.2315 ; 0.1109 |
| 955804 | soup_hybrid_late_mbr16 | 0.8491 ; 0.2345 ; 0.1140 |
| 955815 | H5_hybrid_testmix_mbr16 | 0.8497 ; 0.2361 ; 0.1179 |
| 963010 | **B_anneal_ema_mbr64_20261005 (best)** | 0.8518 ; 0.2477 ; 0.1296 |

## Phase 2 (strict macro F1 ; P ; R)

| Id | Submission | Scores |
| --- | --- | --- |
| 975684 | **P2_E2_dense_lorma (best)**: Dense + LoRMA+, CRF + biaffine, post processing | 0.6749 ; 0.6925 ; 0.6746 |
| 975694 | P2_E1_dense_post: Dense full fine tune + post processing | 0.6025 ; 0.6261 ; 0.5913 |
| 975697 | P2_E1_dense_full: Dense full fine tune | 0.5672 ; 0.6000 ; 0.5540 |

## Phase 3 (macro F1 ; P ; R)

| Id | Submission | Scores |
| --- | --- | --- |
| 989675 | **P3_E1_dense_lorma_dissent (best)**: Dense + LoRMA+, dissent rule | 0.6657 ; 0.7320 ; 0.6328 |
| 989673 | P3_E1_dense_lorma: decoded, no rule | 0.5833 ; 0.6445 ; 0.5559 |
| 989677 | P3_E1_dense_lorma_raw: argmax, no decoder | 0.5818 ; 0.6687 ; 0.5751 |
| 989682 | P3_E2_hybrid_rslora: Hybrid + rsLoRA | 0.5644 ; 0.6140 ; 0.5438 |
| 989685 | P3_E2_hybrid_rslora_dissent | 0.6459 ; 0.6990 ; 0.6208 |
| 989686 | P3_E3_ced_ia3: CED + IA3 | 0.5272 ; 0.5776 ; 0.5022 |
| 989688 | P3_E3_ced_ia3_dissent | 0.5979 ; 0.6441 ; 0.5791 |

## Phase 4 (macro F1 ; P ; R ; D-sel)

| Submission | Scores |
| --- | --- |
| **P4_dense_views_noleak (best)**: Dense + LoRMA+, 2K and 4K tail views | 0.8517 ; 0.8585 ; 0.8438 ; 0.8540 |
| P4_hybrid_adapter_noleak: Hybrid + bottleneck adapters | 0.8411 ; 0.8487 ; 0.8320 ; 0.8590 |
| P4_dense_lorma_noleak: Dense + LoRMA+, single view | 0.8405 ; 0.8457 ; 0.8346 ; 0.8318 |

## Phase 5 (macro F1)

| Submission | Score |
| --- | --- |
| **P5_E2_zlex (best)**: CED decoder head + kNN, blended with TF-IDF SVC margins | 0.4746 |
| P5_E2_ced: CED decoder head + kNN alone | - |
| P5_probe_zlex, P5_probe_zlex_global, P5_probe_blend, P5_probe_blend_dup, P5_probe_knn: probe blends | - |
| P5_tfidf_global, P5_tfidf_k0, P5_tfidf_k50, P5_tfidf_k200, P5_tfidf_lr, P5_tfidf_t030: TF-IDF baselines | - |

## Phase 6 (BERTScore F1 ; ROUGE-1 ; ROUGE-L)

| Submission | Scores |
| --- | --- |
| **H1_mbr32 (best)**: Hybrid + LoRMA+, MBR over 32 samples | 0.8534 ; 0.4837 ; 0.2407 |
| H1_mbr16: same model, MBR over 16 | 0.8533 ; 0.4804 ; 0.2310 |
| P6_hybrid_lorma_full: same model, fixed 1,200 word greedy decode | 0.8507 ; 0.5241 ; 0.2709 |
| P6_hybrid_lorma_p50, P6_hybrid_lorma_p75, P6_hybrid_lorma_drop_lead, P6_hybrid_lorma_drop_tail: length and view variants | - |
| E1_greedy, E1_mbr16: Hybrid full fine tune | - |
| P6_model_200, P6_model_400: short budget models | - |
| P6_extract_*: extractive baselines (lead, tail, lead + tail, cue sentences) | - |

## Phase 7 (F1@7)

| Submission | Score |
| --- | --- |
| **P7_R5_reporter_citator (best)**: LightGBM lambdarank, 39 features, reporter links forced | 0.4944 |
| P7_R2_s200_ltr_citator: lambdarank, 200 candidate shortlist | 0.4832 |
| P7_E0_citator: Okapi BM25 fusion + citator edges | 0.4371 |
| P7_E0: Okapi BM25 fusion, Sinkhorn decoding | 0.4312 |
| P7_E5_citator: learned fusion + citator | 0.4141 |
| P7_bm25, P7_bm25_fuse, P7_bm25_head40, P7_fix_*, P7_struct_*: early lexical baselines | 0.366 to 0.380 |
| P7_R1_ngram, P7_R1_ngram_citator, P7_R2_ltr_citator, P7_R3_ltr_lorma_citator, P7_R5_reporter: n-gram and ranking variants | - |
| P7_rerank, P7_rerank_ce, P7_rerank_lam, P7_rerank_swap7: neural rerankers | - |
