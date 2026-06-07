"""Math tool agent evaluator: checks if the final answer matches ground truth.

Uses rllm's math grading utilities for robust comparison that handles LaTeX,
symbolic expressions, and numeric formats (e.g. ``\\frac{1}{2}`` vs ``0.5``).
"""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.rewards.math_utils.utils import grade_answer_mathd, grade_answer_sympy
from rllm.types import Episode


@rllm.evaluator
def math_tool_evaluator(task: dict, episode: Episode) -> EvalOutput:
    """Grade the agent's answer against ground truth using symbolic math comparison."""
    answer_text = str(episode.artifacts.get("answer", ""))
    # math500 carries the answer in `answer` and the full solution in `ground_truth`;
    # gsm8k and others carry the answer in `ground_truth`. Prefer `answer`.
    ground_truth = str(task.get("answer") or task.get("ground_truth") or "")
    # Check whether there exists tool calls in the episode
    has_tool_call = False
    valid_tool_call = True
    for traj in episode.trajectories:
        if len(traj.steps) == 0:
            continue

        messages = traj.steps[-1].chat_completions
        for msg in messages:
            if msg["role"] == "tool":
                has_tool_call = True
                if "error" in msg["content"].lower():
                    valid_tool_call = False

    is_correct = grade_answer_mathd(answer_text, ground_truth) or grade_answer_sympy(answer_text, ground_truth)
    is_correct = is_correct and has_tool_call and valid_tool_call

    reward = 1.0 if is_correct else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=reward)],
    )
