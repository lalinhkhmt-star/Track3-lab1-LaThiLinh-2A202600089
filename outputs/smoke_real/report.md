# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: real_llm
- Records: 2
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.0 | 0.0 | 0.0 |
| Avg attempts | 1 | 1 | 0 |
| Avg tokens | 455 | 454 | -1 |
| Avg latency (ms) | 5461 | 4392 | -1069 |

## Failure modes
```json
{
  "react": {
    "entity_drift": 1
  },
  "overall": {
    "entity_drift": 2
  },
  "reflexion": {
    "entity_drift": 1
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
