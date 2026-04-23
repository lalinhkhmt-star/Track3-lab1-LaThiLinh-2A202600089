# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: mock
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.5 | 1.0 | 0.5 |
| Avg attempts | 1 | 1.5 | 0.5 |
| Avg tokens | 385 | 790 | 405 |
| Avg latency (ms) | 200 | 455 | 255 |

## Failure modes
```json
{
  "react": {
    "none": 4,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 1,
    "entity_drift": 2
  },
  "reflexion": {
    "none": 8
  },
  "overall": {
    "none": 12,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 1,
    "entity_drift": 2
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- adaptive_max_attempts
- memory_compression
- mock_mode_for_autograding

## Discussion
This benchmark compares a single-pass ReAct baseline with a Reflexion agent that stores concise lessons after failed attempts and feeds the compressed memory into the next actor call. The evaluator uses a structured 0/1 JSON judgment and records failure modes so the report can separate incomplete multi-hop reasoning, entity drift, and wrong final answers. Token totals are taken from API usage fields when a real LLM provider returns them; otherwise the trace explicitly marks a local estimate. Reflexion usually costs more latency and tokens because failed attempts add reflection and retry calls, but it can recover when the first response stops after only one hop or selects an ungrounded second entity.
