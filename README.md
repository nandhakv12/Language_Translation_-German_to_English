# German–English Neural Machine Translation (NMT)

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![NLP](https://img.shields.io/badge/NLP-Seq2Seq-green)
![BLEU](https://img.shields.io/badge/BLEU-34.53-success)

Neural Machine Translation using BiLSTM + Attention  
Graduate-level Deep Learning Project | University at Buffalo

---

## Project Overview

This project implements a German-to-English Neural Machine Translation (NMT) system trained on the Europarl v7 parallel corpus. The system is built from scratch using sequence-to-sequence learning and attention mechanisms. Multiple architectures and optimization strategies are evaluated to understand performance trade-offs under constrained computational budgets.

The final system demonstrates that carefully optimized recurrent architectures can remain competitive with Transformer models in practical settings.

---

## Key Technical Contributions

- Implemented a Bidirectional LSTM encoder–decoder with Luong attention
- Compared recurrent architectures against a Transformer baseline
- Applied SentencePiece Byte Pair Encoding (BPE) for subword tokenization
- Designed confidence-scaled label smoothing to reduce overconfidence
- Implemented AdamW optimization with adaptive gradient clipping
- Evaluated models using BLEU and perplexity metrics

---

## Models and Methods

- **Final Model:** BiLSTM encoder–decoder with attention  
- **Loss Function:** Cross-entropy with label smoothing  
- **Optimizer:** AdamW with weight decay and gradient clipping  
- **Tokenization:** SentencePiece BPE (8,000 vocabulary size)

---

## Results

| Model | Loss | Optimizer | BLEU |
|------|------|-----------|------|
| BiLSTM + Attention | Label Smoothing | AdamW | **34.53** |
| Transformer Baseline | Cross-Entropy | AdamW | 31.5 |

The BiLSTM model achieves higher BLEU than the Transformer baseline under identical training budgets, highlighting the importance of optimization and regularization.

---

## Dataset

- Europarl v7 German–English Parallel Corpus
- Approximately 1.9 million sentence pairs
- Train / Validation / Test split: 80 / 10 / 10

---

## Technology Stack

- Python
- PyTorch
- SentencePiece
- NumPy

---

## Authors

**Nandhakumar Vadivel**  
Master’s in Artificial Intelligence, University at Buffalo  

**Kanisha Raja**  
Master’s in Artificial Intelligence, University at Buffalo

## Repository Usage

```bash
pip install -r requirements.txt
python training/train.py --model bilstm
python evaluation/bleu.py
