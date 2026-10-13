# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Enumerate a graded deterministic tree to audit A-sharp estimators on CPU.

No sampling uncertainty: all eight trajectories and their exact probabilities
are enumerated. Errors concern the raw (unwhitened, unclipped) actor signal.
Run: python -m examples.asharp_trainer.compare_estimators --output /tmp/tree.json
"""

import argparse
import json
import math
from pathlib import Path

import torch

from verl.trainer.ppo.asharp import ACTOR_ESTIMATORS, compute_asharp


def build_tree(beta=1.0, noise_scale=0.0):
    """Return exact moments, perturbed shared state values, and policy scores.

    Seven independent Bernoulli logits parameterize all internal tree states.
    noise_scale perturbs H by a fixed state-consistent pattern; it does not
    inject independent errors into different visits to the same prefix.
    """
    dtype = torch.float64
    right_probability = {1: 0.4, 2: 0.3, 3: 0.65, 4: 0.2, 5: 0.6, 6: 0.75, 7: 0.45}
    leaf_rewards = [-1.0, 0.1, 0.4, 1.0, 1.4, 2.0, 3.0, 4.0]
    h = {}
    for node in reversed(range(1, 8)):
        p = right_probability[node]
        if node >= 4:
            left, right = leaf_rewards[2 * (node - 4) : 2 * (node - 4) + 2]
        else:
            left, right = h[2 * node], h[2 * node + 1]
        if beta == 0:
            h[node] = (1 - p) * left + p * right
        else:
            h[node] = (
                torch.logsumexp(
                    torch.tensor([math.log1p(-p) + beta * left, math.log(p) + beta * right], dtype=dtype), 0
                ).item()
                / beta
            )
    # Includes state-dependent biases in both the baseline and next-state values.
    noise = [0.2, -0.7, 0.5, 1.0, -0.4, 0.8, -0.9]
    exact_values, estimated_values, rewards, probabilities, scores, root_actions = [], [], [], [], [], []
    for leaf in range(8):
        actions = [(leaf >> shift) & 1 for shift in (2, 1, 0)]
        node, probability = 1, 1.0
        exact_row, estimated_row = [], []
        score = torch.zeros(3, 7, dtype=dtype)
        for t, action in enumerate(actions):
            p = right_probability[node]
            probability *= p if action else 1 - p
            exact_row.append(h[node])
            estimated_row.append(h[node] + noise_scale * noise[node - 1])
            score[t, node - 1] = action - p
            node = 2 * node + action
        exact_values.append(exact_row)
        estimated_values.append(estimated_row)
        rewards.append([0.0, 0.0, leaf_rewards[leaf]])
        probabilities.append(probability)
        root_actions.append(actions[0])
        scores.append(score)
    return {
        "rewards": torch.tensor(rewards, dtype=dtype),
        "exact_values": torch.tensor(exact_values, dtype=dtype),
        "values": torch.tensor(estimated_values, dtype=dtype),
        "mask": torch.ones(8, 3, dtype=dtype),
        "probabilities": torch.tensor(probabilities, dtype=dtype),
        "scores": torch.stack(scores),
        "root_actions": torch.tensor(root_actions),
    }


def compare(beta=1.0, lam_actor=0.8, noise_scale=0.0):
    """Root-action conditional bias/variance/MSE and full seven-logit gradient error."""
    tree = build_tree(beta, noise_scale)
    exact, _ = compute_asharp(tree["rewards"], tree["exact_values"], tree["mask"], beta)
    probabilities, scores = tree["probabilities"], tree["scores"]
    exact_gradient = (probabilities[:, None, None] * exact[:, :, None] * scores).sum((0, 1))
    results = {}
    for estimator in ACTOR_ESTIMATORS:
        adv, _ = compute_asharp(
            tree["rewards"],
            tree["values"],
            tree["mask"],
            beta,
            lam_critic=0.95,
            lam_actor=lam_actor,
            actor_estimator=estimator,
        )
        root_metrics = []
        for action in (0, 1):
            select = tree["root_actions"] == action
            weights = probabilities[select] / probabilities[select].sum()
            samples, truth = adv[select, 0], exact[select, 0]
            mean = (weights * samples).sum()
            bias = (weights * (samples - truth)).sum()
            root_metrics.append(
                {
                    "action": action,
                    "bias": bias.item(),
                    "variance": (weights * (samples - mean).square()).sum().item(),
                    "mse": (weights * (samples - truth).square()).sum().item(),
                }
            )
        gradient = (probabilities[:, None, None] * adv[:, :, None] * scores).sum((0, 1))
        results[estimator] = {
            "root_conditional": root_metrics,
            "actor_gradient_l2_error": torch.linalg.vector_norm(gradient - exact_gradient).item(),
        }
    return {"beta": beta, "lam_actor": lam_actor, "noise_scale": noise_scale, "estimators": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "method": "Exact enumeration; raw advantages; seven independent Bernoulli actor logits",
        "cases": [
            compare(beta, lam, noise)
            for beta in (0.0, 0.5, 1.0)
            for lam in (0.0, 0.8, 1.0)
            for noise in (0.0, 0.1, 0.3)
        ],
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
