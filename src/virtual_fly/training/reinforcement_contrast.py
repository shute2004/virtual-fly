from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np


def summarize_effect(delta: np.ndarray, *, epsilon: float = 1e-7) -> dict[str, object]:
    values = np.asarray(delta, dtype=np.float64)
    changed = np.abs(values) > float(epsilon)
    selected = values[changed]
    positive = selected > 0.0
    negative = selected < 0.0
    return {
        "edges": int(values.size),
        "changed_edges": int(changed.sum()),
        "changed_fraction": float(changed.mean()) if values.size else 0.0,
        "positive_edges": int(positive.sum()),
        "negative_edges": int(negative.sum()),
        "mean_signed_delta": float(selected.mean()) if selected.size else 0.0,
        "mean_abs_delta": float(np.abs(selected).mean()) if selected.size else 0.0,
        "max_abs_delta": float(np.abs(selected).max()) if selected.size else 0.0,
        "l1_delta": float(np.abs(values).sum()),
        "l2_delta": float(np.linalg.norm(values)),
    }


def compare_effects(
    reward_delta: np.ndarray,
    aversive_delta: np.ndarray,
    *,
    epsilon: float = 1e-7,
) -> dict[str, object]:
    reward = np.asarray(reward_delta, dtype=np.float64)
    aversive = np.asarray(aversive_delta, dtype=np.float64)
    if reward.shape != aversive.shape:
        raise ValueError("reward and aversive delta arrays must have the same shape")

    reward_changed = np.abs(reward) > float(epsilon)
    aversive_changed = np.abs(aversive) > float(epsilon)
    shared = reward_changed & aversive_changed
    union = reward_changed | aversive_changed
    reward_only = reward_changed & ~aversive_changed
    aversive_only = aversive_changed & ~reward_changed

    shared_reward = reward[shared]
    shared_aversive = aversive[shared]
    same_sign = np.sign(shared_reward) == np.sign(shared_aversive)
    opposite_sign = np.sign(shared_reward) == -np.sign(shared_aversive)

    union_reward = reward[union]
    union_aversive = aversive[union]
    denominator = float(np.linalg.norm(union_reward) * np.linalg.norm(union_aversive))
    cosine = (
        float(np.dot(union_reward, union_aversive) / denominator)
        if denominator > 0.0
        else 0.0
    )

    reward_norm = float(np.linalg.norm(reward))
    aversive_norm = float(np.linalg.norm(aversive))
    distance = float(np.linalg.norm(reward - aversive))
    normalization = reward_norm + aversive_norm

    return {
        "edges": int(reward.size),
        "reward": summarize_effect(reward, epsilon=epsilon),
        "aversive": summarize_effect(aversive, epsilon=epsilon),
        "shared_changed_edges": int(shared.sum()),
        "reward_only_edges": int(reward_only.sum()),
        "aversive_only_edges": int(aversive_only.sum()),
        "union_changed_edges": int(union.sum()),
        "changed_jaccard": (
            float(shared.sum() / union.sum()) if int(union.sum()) > 0 else 0.0
        ),
        "shared_same_sign_edges": int(same_sign.sum()),
        "shared_opposite_sign_edges": int(opposite_sign.sum()),
        "shared_same_sign_fraction": (
            float(same_sign.mean()) if shared_reward.size else 0.0
        ),
        "shared_opposite_sign_fraction": (
            float(opposite_sign.mean()) if shared_reward.size else 0.0
        ),
        "cosine_similarity_on_union": cosine,
        "l2_distance": distance,
        "normalized_l2_distance": distance / normalization if normalization > 0.0 else 0.0,
    }


def comparison_from_transaction_contrast(response: dict[str, object]) -> dict[str, object]:
    plastic_edges = int(response["plastic_edges"])
    reward_raw = dict(response["reward"])
    aversive_raw = dict(response["aversive"])

    def normalize_effect(raw: dict[str, object], norm_sq: float) -> dict[str, object]:
        shift_changed = int(raw["shift_changed_edges"])
        changed = int(raw["changed_edges"])
        return {
            "edges": plastic_edges,
            "changed_edges": changed,
            "changed_fraction": changed / plastic_edges if plastic_edges else 0.0,
            "shift_changed_edges": shift_changed,
            "bound_changed_edges": int(raw["bound_changed_edges"]),
            "positive_edges": int(raw["positive_shift_edges"]),
            "negative_edges": int(raw["negative_shift_edges"]),
            "mean_signed_delta": (
                float(raw["sum_shift_delta"]) / shift_changed if shift_changed else 0.0
            ),
            "mean_abs_delta": (
                float(raw["sum_abs_shift_delta"]) / shift_changed if shift_changed else 0.0
            ),
            "max_abs_delta": float(raw["max_abs_shift_delta"]),
            "l1_delta": float(raw["sum_abs_shift_delta"]),
            "l2_delta": sqrt(max(0.0, norm_sq)),
        }

    reward_norm_sq = float(response["reward_shift_norm_sq"])
    aversive_norm_sq = float(response["aversive_shift_norm_sq"])
    dot = float(response["shift_dot"])
    shared = int(response["shared_changed_edges"])
    reward_only = int(response["reward_only_changed_edges"])
    aversive_only = int(response["aversive_only_changed_edges"])
    union = shared + reward_only + aversive_only
    shared_shift = int(response["shared_shift_edges"])
    same_sign = int(response["shared_same_sign_shift_edges"])
    opposite_sign = int(response["shared_opposite_sign_shift_edges"])
    denominator = sqrt(max(0.0, reward_norm_sq * aversive_norm_sq))
    distance_sq = max(0.0, reward_norm_sq + aversive_norm_sq - 2.0 * dot)
    reward_norm = sqrt(max(0.0, reward_norm_sq))
    aversive_norm = sqrt(max(0.0, aversive_norm_sq))
    normalization = reward_norm + aversive_norm

    return {
        "edges": plastic_edges,
        "reward": normalize_effect(reward_raw, reward_norm_sq),
        "aversive": normalize_effect(aversive_raw, aversive_norm_sq),
        "shared_changed_edges": shared,
        "reward_only_edges": reward_only,
        "aversive_only_edges": aversive_only,
        "union_changed_edges": union,
        "changed_jaccard": shared / union if union else 0.0,
        "shared_same_sign_edges": same_sign,
        "shared_opposite_sign_edges": opposite_sign,
        "shared_same_sign_fraction": same_sign / shared_shift if shared_shift else 0.0,
        "shared_opposite_sign_fraction": opposite_sign / shared_shift if shared_shift else 0.0,
        "cosine_similarity_on_union": dot / denominator if denominator > 0.0 else 0.0,
        "l2_distance": sqrt(distance_sq),
        "normalized_l2_distance": sqrt(distance_sq) / normalization if normalization > 0.0 else 0.0,
    }


def render_markdown(payload: dict[str, object]) -> str:
    comparison = dict(payload["comparison"])
    reward = dict(comparison["reward"])
    aversive = dict(comparison["aversive"])
    lines = [
        "# Flyppy reinforcement contrast probe",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- checkpoint global weight version: {payload['global_weight_version']}",
        f"- fixed initial condition: `{payload['condition']}`",
        f"- course seed: {payload.get('course_seed', '-')}",
        f"- replayed sensory frames: {payload.get('sensory_frames', 0)}",
        f"- captured trajectory event: `{dict(payload.get('trajectory_event') or {}).get('reason', 'unknown')}`",
        f"- reinforcement steps: {payload['reinforcement_steps']}",
        f"- reward / aversive current: {payload['reward_current']} / {payload['aversive_current']}",
        f"- epsilon: {payload['epsilon']}",
        "",
        "## Event-isolated weight effect",
        "",
        "control / reward / aversive の3 slotへ同じclosed-loop sensory current列をreplayし、最後のreinforcement stepだけ分岐させたtransaction差を比較する。",
        "",
        "| metric | reward | aversive |",
        "|---|---:|---:|",
        f"| changed edges | {reward['changed_edges']} | {aversive['changed_edges']} |",
        f"| positive edges | {reward['positive_edges']} | {aversive['positive_edges']} |",
        f"| negative edges | {reward['negative_edges']} | {aversive['negative_edges']} |",
        f"| mean signed delta | {float(reward['mean_signed_delta']):.9g} | {float(aversive['mean_signed_delta']):.9g} |",
        f"| mean abs delta | {float(reward['mean_abs_delta']):.9g} | {float(aversive['mean_abs_delta']):.9g} |",
        f"| max abs delta | {float(reward['max_abs_delta']):.9g} | {float(aversive['max_abs_delta']):.9g} |",
        "",
        "## Reward / aversive contrast",
        "",
        f"- shared changed edges: {comparison['shared_changed_edges']}",
        f"- reward-only edges: {comparison['reward_only_edges']}",
        f"- aversive-only edges: {comparison['aversive_only_edges']}",
        f"- changed-edge Jaccard: {float(comparison['changed_jaccard']):.3%}",
        f"- shared same-sign fraction: {float(comparison['shared_same_sign_fraction']):.3%}",
        f"- shared opposite-sign fraction: {float(comparison['shared_opposite_sign_fraction']):.3%}",
        f"- cosine similarity on union: {float(comparison['cosine_similarity_on_union']):.6f}",
        f"- normalized L2 distance: {float(comparison['normalized_l2_distance']):.6f}",
        "",
        "## Interpretation boundary",
        "",
        "これは固定初期条件からproduction CNS→motor→FlyBody閉ループを実際に走らせ、最初のgate pass / collision / finishまでに生じた実R1-R6感覚入力履歴を3 CNS slotへ同一replayした診断である。",
        "event直前までのeligibility履歴を揃えた上で、PAM01またはPPL101刺激が現在の局所plasticity transactionへ与える差だけを切り分ける。",
        "外部reward符号やtarget weightは使用しない。",
        "",
    ]
    return "\n".join(lines)
