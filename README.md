# touchstone

**Bias-aware LLM judge with multi-model consensus and calibration drift detection.**

```bash
pip install touchstone-llm[anthropic]
```

```python
from touchstone import Judge, HELPFULNESS

judge = Judge(model="claude-sonnet-4-6")
result = await judge.score(output, rubric=HELPFULNESS)

print(result.adjusted_score)   # bias-corrected score
print(result.bias_flags)       # ["verbosity"] if flagged
print(result.passed)           # True if above threshold
```

---

## Why touchstone?

Every LLM eval framework runs a judge prompt and returns a score. Nobody corrects for what happens before that score reaches you.

LLM judges have three systematic biases that inflate or deflect scores silently:

| Bias | What happens | touchstone fix |
|------|-------------|---------------|
| **Verbosity** | Longer outputs score higher regardless of quality | Length-normalized penalty |
| **Position** | In A/B comparisons, first answer wins more often | Runs both orderings, reconciles |
| **Self-preference** | Claude scores Claude-generated text higher | Detected and flagged automatically |

Plus: when two judges disagree, touchstone tells you — instead of silently averaging.

---

## Features

- **Single judge** — score any output against a composable rubric
- **Pairwise comparison** — A vs B with position-bias correction built in
- **Panel judging** — run N models in parallel, compute agreement score, flag disputes
- **Bias correction** — verbosity, position, and self-preference detection out of the box
- **Calibration drift** — baseline your judge, detect silent model updates
- **Provider-agnostic** — Anthropic, OpenAI, Gemini, Ollama, any OpenAI-compatible endpoint
- **CLI** — `touchstone score`, `touchstone compare`, `touchstone panel` for CI/CD

---

## Quick examples

**Score with custom rubric:**
```python
from touchstone import Judge, Rubric

rubric = (
    Rubric()
    .add("factual accuracy", weight=0.5)
    .add("conciseness", weight=0.3)
    .add("tone", weight=0.2)
)

result = await judge.score(my_output, rubric=rubric)
```

**Panel with consensus:**
```python
from touchstone import Panel

panel = Panel(models=["claude-sonnet-4-6", "gpt-4o", "gemini-2.0-flash"])
result = await panel.evaluate(output, rubric=["accuracy", "clarity"])

print(result.consensus_score)  # 0.84
print(result.agreement)        # 0.91
print(result.disputed)         # False
```

**Pairwise comparison:**
```python
result = await judge.compare(output_a, output_b, rubric=["helpfulness"])
print(result.winner)                   # "a"
print(result.position_bias_detected)   # False
```

**Calibration drift:**
```python
from touchstone import Calibration

cal = Calibration(judge)
await cal.baseline(reference_samples, rubric)  # run once

report = await cal.check(reference_samples, rubric)
print(report.drift_score)  # 0.03 = stable, >0.05 = drifted
```

**CLI / CI gate:**
```bash
touchstone score "Paris is the capital of France." -c "factual accuracy" --threshold 0.8
# exits 1 if below threshold
```

---

## Installation

```bash
# Anthropic
pip install touchstone-llm[anthropic]

# OpenAI
pip install touchstone-llm[openai]

# Both + CLI
pip install touchstone-llm[all]

# Local models (Ollama)
pip install touchstone-llm[ollama]
```

---

## Built-in rubrics

```python
from touchstone import FACTUALITY, HELPFULNESS, SAFETY, RAG_QUALITY
```

---

## License

MIT
