# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: real_llm
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.0 | 0.06 | 0.06 |
| Avg attempts | 1 | 2.96 | 1.96 |
| Avg tokens | 506.31 | 2334.79 | 1828.48 |
| Avg latency (ms) | 3781.81 | 14579.68 | 10797.87 |

## Failure modes
```json
{
  "react": {
    "entity_drift": 100
  },
  "overall": {
    "entity_drift": 100,
    "reflection_overfit": 94,
    "none": 6
  },
  "reflexion": {
    "reflection_overfit": 94,
    "none": 6
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- adaptive_max_attempts
- memory_compression
- actual_api_token_usage

## Discussion
This benchmark compares a single-pass ReAct baseline with a Reflexion agent that stores concise lessons after failed attempts and feeds the compressed memory into the next actor call. The evaluator returns a structured 0/1 judgment while the final EM score is checked with normalized gold-answer matching, so the report can separate incomplete multi-hop reasoning, entity drift, and wrong final answers. Token totals are taken from provider usage fields when the LLM API returns them; if a provider omits usage, each trace marks the local fallback explicitly. Reflexion costs more latency and tokens when it needs retries, but it can recover from answers that stop after only one hop or drift to an ungrounded entity.
