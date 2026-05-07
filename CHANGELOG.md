# Changelog

## [0.1.0] - 2026-05-07

### Added
- `Judge` — single-model scoring with bias detection and correction
- `Panel` — multi-model parallel evaluation with inter-judge agreement scoring
- `Rubric` — composable, weighted evaluation criteria builder
- Built-in rubrics: `FACTUALITY`, `HELPFULNESS`, `SAFETY`, `RAG_QUALITY`
- Bias correction: verbosity normalization, position bias in pairwise, self-preference detection
- `Calibration` — baseline and drift detection for silent model updates
- Provider support: Anthropic, OpenAI, Gemini (via OpenAI-compat), Ollama
- CLI: `touchstone score`, `touchstone compare`, `touchstone panel`
- `ComparisonResult` with `position_bias_detected` flag
