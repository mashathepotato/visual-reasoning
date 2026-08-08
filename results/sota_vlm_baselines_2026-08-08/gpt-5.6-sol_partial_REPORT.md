# Direct VLM baseline: gpt-5.6-sol

Reasoning effort: `high`. Protocol fingerprint: `96ce3c1e2bb90b6466750517838daa8893e7cd0ce3626c9b6ebcf9880924bca8`.
Complete: `False` (279/711 responses).

| Task | Metric | cached n | expected n | Value | 95% CI | Invalid | Complete |
|---|---|---:|---:|---:|---:|---:|---:|
| tetris_ood | accuracy | 100 | 100 | 0.7200 | [0.6251, 0.7986] | 0 | True |
| colored_ood | accuracy | 100 | 100 | 0.9800 | [0.9300, 0.9945] | 0 | True |
| ganis3d | accuracy | 75 | 78 | 0.5733 | [0.4605, 0.6790] | 0 | False |
| maze_trace | accuracy | 4 | 100 | 0.7500 | [0.3006, 0.9544] | 0 | False |

Token usage: {"cached_input_tokens": 0, "input_tokens": 67886, "output_tokens": 460190, "reasoning_tokens": 457836}.
Estimated list-price cost: $14.1451.
