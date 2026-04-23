from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .mock_runtime import actor_answer, classify_failure, evaluator, reflector
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_judge = JudgeResult(score=0, reason="No attempts were run.")

        for attempt_id in range(1, self.max_attempts + 1):
            actor_step = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            answer = actor_step.value
            judge_step = evaluator(example, answer)
            judge = judge_step.value

            prompt_tokens = actor_step.prompt_tokens + judge_step.prompt_tokens
            completion_tokens = actor_step.completion_tokens + judge_step.completion_tokens
            token_total = actor_step.token_estimate + judge_step.token_estimate
            latency_ms = actor_step.latency_ms + judge_step.latency_ms
            token_source = (
                "api_usage"
                if actor_step.token_source == "api_usage" and judge_step.token_source == "api_usage"
                else "local_estimate"
            )

            reflection = None
            final_answer = answer
            final_score = judge.score
            final_judge = judge

            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection_step = reflector(example, attempt_id, judge, answer)
                reflection = reflection_step.value
                reflections.append(reflection)
                reflection_memory.append(
                    f"Attempt {attempt_id}: {reflection.lesson} Next strategy: {reflection.next_strategy}"
                )
                prompt_tokens += reflection_step.prompt_tokens
                completion_tokens += reflection_step.completion_tokens
                token_total += reflection_step.token_estimate
                latency_ms += reflection_step.latency_ms
                if reflection_step.token_source != "api_usage":
                    token_source = "local_estimate"

            traces.append(
                AttemptTrace(
                    attempt_id=attempt_id,
                    answer=answer,
                    score=judge.score,
                    reason=judge.reason,
                    reflection=reflection,
                    token_estimate=token_total,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    token_source=token_source,
                    latency_ms=latency_ms,
                )
            )

            if judge.score == 1:
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = classify_failure(
            example,
            final_answer,
            judge=final_judge,
            attempts=len(traces),
        )
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
