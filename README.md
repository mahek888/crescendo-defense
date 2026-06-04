# CrescendoDefense

A three-layer runtime defense framework for detecting and mitigating Crescendo-style multi-turn jailbreak attacks in Large Language Models (LLMs).

## Overview

CrescendoDefense is a lightweight, model-agnostic runtime security framework designed to defend against conversational jailbreak attacks that gradually escalate toward harmful objectives over multiple turns.

Unlike traditional prompt-level moderation systems, CrescendoDefense analyzes the evolving conversational trajectory, selectively disrupts adversarial context accumulation, and audits generated responses before delivery.

The framework specifically targets the four primary mechanisms exploited by Crescendo-style attacks:

- Memory Stacking
- Guard-Lowering Dialogue
- Semantic Drift
- Prompt Disguising

---

## Architecture

The framework consists of three complementary defense layers:

### Layer 1: Semantic Kinematic Detector

Monitors conversational trajectories using four semantic risk signals:

- Absolute Risk (D)
- Semantic Velocity (V)
- Semantic Acceleration (A)
- Cumulative Risk (C)

Detects gradual escalation patterns characteristic of Crescendo attacks.

### Layer 2: Strategic Context Eviction

Activated when Layer 1 detects suspicious escalation.

Removes intermediate conversational scaffolding while preserving key contextual anchors:

```
Compressed Context =
[System Prompt]
+ [First User Turn]
+ [Previous User Turn]
+ [Latest User Turn]
```

Disrupts adversarial memory accumulation and guard-lowering dialogue.

### Layer 3: Semantic Response Auditor

Audits generated response prefixes before delivery.

Uses semantic similarity against unsafe completion profiles to detect:

- Malware assistance
- Phishing assistance
- Cyber exploitation
- PII harvesting
- Chemical hazards
- Criminal facilitation
- Other restricted content

---

## Experimental Setup

### Target Model

- Llama-3.2-3B-Instruct

### Embedding Model

- all-MiniLM-L6-v2

### Benchmark

22 multi-turn conversational scenarios:

| Type | Count |
|--------|--------:|
| Adversarial | 15 |
| Benign | 5 |
| Risky-Benign | 2 |
| Total | 22 |

---

## Results

| Configuration | ASR (%) | FPR (%) |
|---------------|---------|---------|
| Raw Model | 86.67 | 0.00 |
| Layer 1 + Layer 2 | 40.00 | 0.00 |
| Layer 1 + Layer 3 | 33.33 | 28.57 |
| Full Pipeline | 26.67 | 28.57 |

### Key Findings

- Reduced ASR from **86.67%** to **26.67%**
- Achieved a **69.2% relative reduction** in successful jailbreak attacks
- Layer 1 + Layer 2 maintained **0% False Positive Rate**
- Demonstrated effectiveness of layered runtime defenses against multi-turn jailbreak attacks

---

## Repository Structure

```text
src/
├── config.py
├── pipeline.py
├── layer1_detector.py
├── layer2_eviction.py
├── layer3_verifier.py

dataset/
docs/

benchmark.py
benchmark_protected.py
requirements.txt
```

---

## Report

The complete technical report and appendix are available in:

```text
docs/
```

---

## Future Work

Potential future extensions include:

- Adaptive thresholding
- Dynamic safety anchor generation
- Improved context-retention strategies
- Integration with PyRIT and Giskard
- Evaluation on larger benchmark suites and additional LLMs

---

## Author

**Mahek Nishant Vedant**

B.Tech Computer Science & Engineering  
Delhi Technological University

---


