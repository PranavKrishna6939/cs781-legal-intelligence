"""Saul: CS-781 Capstone Phase-1 legal foundation-model pipeline.

Sub-packages
------------
saul.data       streaming preprocessing, dedup, anonymization, splits, packed loader
saul.tokenizer  legal byte-level BPE training, release gates, corpus tokenization
saul.model      model components and HF-compatible SaulForCausalLM (dense/hybrid/ced/conv)
saul.train      Muon+AdamW optimizer, WSD schedule, EMA, trainer, distillation
saul.eval       official-metric wrappers, checkpoint selection, paired bootstrap
saul.decode     generation engine, logits processors, contrastive search, MBR, submission
"""

__version__ = "0.1.0"
