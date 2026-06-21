from collections import deque

import numpy as np
import torch


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.1):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.all_time_actions = np.zeros([220 + 80, 220 * 7 + 7, 7])

    def reset(self):
        self.action_history.clear()
        self.all_time_actions = np.zeros([220 + 80, 220 * 7 + 7, 7])

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        ref = curr_act_preds[num_actions - 1, :]
        dot_product = np.sum(curr_act_preds * ref, axis=1)
        norm_previous_pred = np.linalg.norm(curr_act_preds, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * curr_act_preds, axis=0)

    def ensemble_action_c2f_cosine_sim(self, cont_cur_action, discrete_cur_action):
        if discrete_cur_action.ndim == 1:
            discrete_cur_action = discrete_cur_action[None, :]
            cont_cur_action = cont_cur_action[None, :]

        dot_product = np.sum(discrete_cur_action * cont_cur_action, axis=1)
        norm_discrete = np.linalg.norm(discrete_cur_action, axis=1)
        norm_continuous = np.linalg.norm(cont_cur_action, axis=1)
        cos_similarity = dot_product / (norm_discrete * norm_continuous + 1e-7)

        denom = np.exp(self.adaptive_ensemble_alpha * cos_similarity) + np.exp(
            self.adaptive_ensemble_alpha * (1 - cos_similarity)
        )
        weights = (np.exp(self.adaptive_ensemble_alpha * cos_similarity) / denom)[:, None]
        return weights * cont_cur_action + (1 - weights) * discrete_cur_action

    def blend_l1_weighted(self, continuous_actions, discrete_actions, alpha=5.0):
        """Blend actions with element-wise exponential decay on L1 distance."""
        l1_dist = np.abs(continuous_actions - discrete_actions)
        weights = np.exp(-alpha * l1_dist)
        print("weights: ", weights)
        return weights * continuous_actions + (1 - weights) * discrete_actions

    def conditional_l1_blending(
        self,
        continuous_actions,
        discrete_actions,
        discrete_confidence,
        alpha=5.0,
        confidence_threshold=0.9,
    ):
        """
        Blend continuous and discrete actions unless discrete confidence is high
        enough to fully trust the discrete branch.
        """
        discrete_confidence = discrete_confidence.cpu().numpy()

        l1_dist = np.abs(continuous_actions - discrete_actions)
        l1_weights = np.exp(-alpha * l1_dist)
        discrete_trust_mask = (discrete_confidence >= confidence_threshold).astype(float)
        combined_weights = l1_weights * (1 - discrete_trust_mask)

        return combined_weights * continuous_actions + (1 - combined_weights) * discrete_actions

    def temp_ensemble(self, discrete_actions, t):
        all_actions = discrete_actions
        num_queries = all_actions.shape[0]
        self.all_time_actions[t, t : t + num_queries, :] = all_actions

        actions_for_curr_step = self.all_time_actions[:, t, :]
        mask = np.any(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[mask]

        k = 0.01
        n = actions_for_curr_step.shape[0]
        weights = np.exp(-k * np.arange(n))
        weights /= weights.sum()
        weights = weights[:, np.newaxis]

        return np.sum(actions_for_curr_step * weights, axis=0, keepdims=True)


if __name__ == "__main__":
    continuous_actions = np.array([
        [0.055, -0.016, 0.182, -0.066, 0.023, -0.007, 0.002],
        [0.064, -0.008, 0.328, -0.071, 0.008, -0.012, 0.001],
        [0.043, 0.016, 0.454, -0.083, 0.004, -0.011, 0.002],
        [0.030, 0.064, 0.622, -0.091, 0.010, -0.013, 0.004],
        [0.025, 0.119, 0.786, -0.089, 0.010, -0.019, 0.007],
        [0.017, 0.195, 0.819, -0.072, 0.009, -0.020, 0.012],
        [0.027, 0.270, 0.870, -0.052, 0.003, -0.013, 0.012],
        [0.027, 0.419, 0.845, -0.056, -0.001, -0.004, 0.012],
    ])

    discrete_actions = np.array([
        [-0.003, -0.001, 0.291, -0.065, 0.039, -0.005, 0.000],
        [-0.003, -0.001, 0.452, -0.073, 0.027, -0.017, 0.000],
        [-0.003, -0.001, 0.547, -0.077, 0.000, 0.000, 0.000],
        [-0.003, 0.065, 0.723, -0.094, 0.000, -0.023, 0.000],
        [-0.003, 0.083, 0.862, -0.083, 0.021, -0.030, 0.000],
        [-0.003, 0.216, 0.862, -0.066, 0.000, -0.022, 0.000],
        [-0.003, 0.330, 0.914, -0.048, 0.000, -0.012, 0.000],
        [-0.003, 0.457, 0.914, -0.051, -0.007, -0.012, 0.000],
    ])

    confidence = torch.tensor([
        [0.9907, 0.9999, 0.3411, 0.7493, 0.2439, 0.4608, 1.0000],
        [0.9519, 1.0000, 0.9900, 0.4178, 0.4882, 0.6749, 1.0000],
        [0.9406, 0.6124, 0.4565, 0.2256, 0.9855, 0.8159, 1.0000],
        [0.9959, 0.6196, 0.7705, 0.4572, 0.6810, 0.5347, 1.0000],
        [0.9999, 0.4911, 0.6166, 0.5331, 0.4866, 0.9489, 1.0000],
        [0.9999, 0.5601, 0.9258, 0.4770, 0.6029, 0.8173, 1.0000],
        [0.9999, 0.8091, 0.3120, 0.7056, 0.9966, 0.9995, 1.0000],
        [0.9959, 0.8393, 0.4031, 0.7712, 0.7982, 0.9980, 1.0000],
    ])

    ada = AdaptiveEnsembler(pred_action_horizon=8, adaptive_ensemble_alpha=10.0)
    smoothed_action_1 = ada.temp_ensemble(discrete_actions[:4], 0)
    smoothed_action_2 = ada.temp_ensemble(discrete_actions[4:], 1)

    print("smoothed_actions_1: ", smoothed_action_1)
    print("smoothed_actions_2: ", smoothed_action_2)
