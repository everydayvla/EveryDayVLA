from collections import defaultdict, deque

import numpy as np


def _mad_per_timestep(continuous_action, discrete_action):
    return np.mean(np.abs(continuous_action - discrete_action), axis=1)


def select_action_by_mad(continuous_action, discrete_action, threshold=0.03):
    """Select a single action branch using continuous/discrete disagreement."""
    mad = np.mean(np.abs(continuous_action - discrete_action))
    print(f"mad: {mad}")
    if mad < threshold:
        return continuous_action
    return discrete_action


def adapt_horizon(continuous_action, discrete_action, threshold=0.0245, min_actions=4):
    """Return the discrete action prefix whose disagreement stays below threshold."""
    mad = _mad_per_timestep(continuous_action, discrete_action)
    print("MAD per timestep:", mad)

    horizon = min_actions
    for i in range(min_actions, len(mad)):
        if mad[i] >= threshold:
            break
        horizon = i + 1

    return discrete_action[:horizon]


class AdaptiveHorizon:
    def __init__(
        self,
        min_actions=4,
        threshold=0.03,
        replan_threshold=0.06,
        max_replan_count=8,
        next_task_thresh=2,
        num_task_mad=3,
        history_len=50,
    ):
        self.max_replan_count = max_replan_count
        self.threshold = threshold
        self.replan_threshold = replan_threshold
        self.next_task_thresh = next_task_thresh
        self.replan_ctr = 0
        self.max_replan_ctr = 0
        self.min_actions = min_actions

        self.num_task_mad_ctr = 0
        self.num_task_mad_thresh = num_task_mad
        self.mad_flag = False
        self.exceed = False

        self.history_len = history_len
        self.action_history = []
        self.mad_history = deque(maxlen=history_len)
        self.calibrator = Calibrator()

    def reset(self):
        self.replan_ctr = 0
        self.mad_flag = False
        self.action_history.clear()
        self.mad_history.clear()

    def full_reset(self):
        self.replan_ctr = 0
        self.max_replan_ctr = 0
        self.mad_flag = False
        self.num_task_mad_ctr = 0
        self.action_history.clear()
        self.mad_history.clear()

    def calibrate(self, task_name):
        self.threshold, self.replan_threshold = self.calibrator.get_thresholds(task_name)

    def record(self, task_name, mad, success):
        self.calibrator.record_episode(task_name, mad, success)

    def _prefix_by_threshold(self, mad, discrete_action, threshold):
        horizon = self.min_actions
        for i in range(self.min_actions, len(mad)):
            if mad[i] >= threshold:
                break
            horizon = i + 1
        return discrete_action[:horizon]

    def adapt_horizon_dplan_beta(self, continuous_action, discrete_action):
        self.exceed = False
        curr_act_preds = self.action_history[0] if self.action_history else discrete_action[::-1]

        mad = _mad_per_timestep(continuous_action, discrete_action)
        print("MAD per timestep:", mad)

        if np.any(mad > self.replan_threshold) and self.min_actions > 1:
            print("Replan triggered due to high MAD.")
            self.replan_ctr += 1
            self.exceed = True

        self.max_replan_ctr = max(self.max_replan_ctr, self.replan_ctr)

        if self.exceed:
            print("Executing fallback action.")
            last_action = curr_act_preds[-1] if curr_act_preds.ndim == 2 else curr_act_preds
            delta = np.array([1e-3] + [0] * (last_action.shape[0] - 1))
            fallback = last_action[None, :] + delta
            self.action_history.append(fallback)
            return fallback, mad

        selected = self._prefix_by_threshold(mad, discrete_action, self.threshold)
        self.action_history.clear()
        self.action_history.append(selected)
        return selected, mad

    def adapt_horizon_dplan_old(self, continuous_action, discrete_action):
        mad = _mad_per_timestep(continuous_action, discrete_action)
        print("MAD per timestep:", mad)

        if np.any(mad[: self.min_actions] > self.replan_threshold) and self.min_actions > 1:
            print("Replan triggered due to high MAD in first {} steps.".format(self.min_actions))
            self.replan_ctr += 1

        self.max_replan_ctr = max(self.max_replan_ctr, self.replan_ctr)

        if self.replan_ctr >= self.next_task_thresh and not self.mad_flag:
            self.num_task_mad_ctr += 1
            self.mad_flag = True

        if (
            self.max_replan_ctr >= self.max_replan_count
            and self.replan_ctr >= self.next_task_thresh
        ) or self.num_task_mad_ctr > self.num_task_mad_thresh:
            return discrete_action

        return self._prefix_by_threshold(mad, discrete_action, self.threshold)

    def adapt_horizon_dplan(self, continuous_action, discrete_action):
        mad = _mad_per_timestep(continuous_action, discrete_action)
        print("MAD per timestep:", mad)

        if np.any(mad[: self.min_actions] > self.replan_threshold) and self.min_actions > 1:
            print("Replan triggered due to high MAD in first {} steps.".format(self.min_actions))
            self.replan_ctr += 1

        self.max_replan_ctr = max(self.max_replan_ctr, self.replan_ctr)

        if self.max_replan_ctr >= self.max_replan_count and self.replan_ctr >= self.next_task_thresh:
            return discrete_action, mad

        return self._prefix_by_threshold(mad, discrete_action, self.threshold), mad

    def switch_actions(self, continuous_action, discrete_action):
        mad = _mad_per_timestep(continuous_action, discrete_action)
        print("MAD per timestep:", mad)
        if np.any(mad > self.replan_threshold):
            return discrete_action
        return continuous_action

    def adapt_horizon_dplan_hist(self, continuous_action, discrete_action):
        mad = _mad_per_timestep(continuous_action, discrete_action)
        print("MAD per timestep:", mad)

        if not self.mad_history:
            self.mad_history.extend([mad] * self.mad_history.maxlen)
        self.mad_history.append(mad)

        running_mad_avg = np.mean(self.mad_history, axis=0)
        mad_std = np.std(self.mad_history, axis=0)
        k = 3
        threshold = running_mad_avg + k * mad_std

        print(f"Using {k} * mad_std, history length: {self.history_len}")

        if np.all(mad < threshold):
            return discrete_action, mad

        print("Replan triggered due to high MAD.")
        return self._prefix_by_threshold(mad, discrete_action, threshold), mad


def adapt_horizon_with_replan(
    continuous_action,
    discrete_action,
    threshold=0.03,
    replan_threshold=0.06,
    min_actions=4,
):
    """Adaptive horizon selection with a replan safeguard."""
    mad = _mad_per_timestep(continuous_action, discrete_action)
    print("MAD per timestep:", mad)

    if np.any(mad[:min_actions] > replan_threshold) and min_actions > 1:
        print("Replan triggered due to high MAD in first {} steps.".format(min_actions))
        return discrete_action

    horizon = min_actions
    for i in range(min_actions, len(mad)):
        if mad[i] >= threshold:
            break
        horizon = i + 1

    return discrete_action[:horizon]


def adapt_horizon_mask(continuous_action, discrete_action, threshold=0.03):
    mad = _mad_per_timestep(continuous_action, discrete_action)
    print("MAD per timestep:", mad)

    mask = mad < threshold
    print("Mask (1 indicates MAD is below threshold):", mask.astype(np.int32))

    if np.any(mask):
        return discrete_action[mask]
    return discrete_action


class Calibrator:
    def __init__(
        self,
        default_threshold=0.03,
        default_replan_threshold=0.06,
        min_actions=2,
        bin_width=0.0078,
        normalize_by_bin=False,
    ):
        self.task_thresholds = {}
        self.mad_history = defaultdict(list)
        self.success_history = defaultdict(list)
        self.default_threshold = default_threshold
        self.default_replan_threshold = default_replan_threshold
        self.min_actions = min_actions
        self.bin_width = bin_width
        self.normalize_by_bin = normalize_by_bin

    def record_episode(self, task_name, mad_values, success):
        """Record MAD values and success status for an episode."""
        self.mad_history[task_name].append(mad_values)
        self.success_history[task_name].append(success)
        print("length of mad_history: ", len(self.mad_history[task_name]))

    def calibrate_task(self, task_name, percentile_threshold=75, percentile_replan=90, min_successful_episodes=2):
        """Calibrate thresholds for a specific task based on historical data."""
        if task_name not in self.mad_history or len(self.mad_history[task_name]) < min_successful_episodes:
            return self.default_threshold, self.default_replan_threshold

        success_indices = [i for i, success in enumerate(self.success_history[task_name]) if success]
        if len(success_indices) < min_successful_episodes:
            return self.default_threshold, self.default_replan_threshold

        success_mad_values = [self.mad_history[task_name][i] for i in success_indices]
        all_mad = np.concatenate(success_mad_values)

        threshold = np.percentile(all_mad, percentile_threshold)
        replan_threshold = np.percentile(all_mad, percentile_replan)

        if self.normalize_by_bin:
            threshold /= self.bin_width
            replan_threshold /= self.bin_width

        self.task_thresholds[task_name] = {
            "threshold": threshold,
            "replan_threshold": replan_threshold,
        }

        return threshold, replan_threshold

    def calibrate_all_tasks(self):
        """Calibrate thresholds for all tasks with sufficient data."""
        for task_name in self.mad_history.keys():
            self.calibrate_task(task_name)

    def get_thresholds(self, task_name):
        """Get calibrated thresholds for a task, or defaults if not calibrated."""
        if task_name in self.task_thresholds:
            return self.task_thresholds[task_name]["threshold"], self.task_thresholds[task_name]["replan_threshold"]
        if task_name in self.mad_history:
            return self.calibrate_task(task_name)
        return self.default_threshold, self.default_replan_threshold

    def dimension_specific_analysis(self, task_name, top_n_dimensions=3):
        """Analyze which dimensions contribute most to MAD in failed episodes."""
        if task_name not in self.mad_history:
            return None

        success_indices = [i for i, success in enumerate(self.success_history[task_name]) if success]
        failure_indices = [i for i, success in enumerate(self.success_history[task_name]) if not success]

        if not failure_indices or not success_indices:
            return None

        return {"analysis": "Dimension-specific analysis requires per-dimension MAD values"}


if __name__ == "__main__":
    continuous_action = np.array([
        [0.055, -0.016, 0.182, -0.066, 0.023, -0.007, 0.002],
        [0.064, -0.008, 0.328, -0.071, 0.008, -0.012, 0.001],
        [0.043, 0.016, 0.454, -0.083, 0.004, -0.011, 0.002],
        [0.030, 0.064, 0.622, -0.091, 0.010, -0.013, 0.004],
        [0.025, 0.119, 0.786, -0.089, 0.010, -0.019, 0.007],
        [0.017, 0.195, 0.819, -0.072, 0.009, -0.020, 0.012],
        [0.027, 0.270, 0.870, -0.052, 0.003, -0.013, 0.012],
        [0.027, 0.419, 0.845, -0.056, -0.001, -0.004, 0.012],
    ])
    discrete_action = np.array([
        [-0.003, -0.001, 0.291, -0.065, 0.039, -0.005, 0.000],
        [-0.003, -0.001, 0.452, -0.073, 0.027, -0.017, 0.000],
        [-0.003, -0.001, 0.547, -0.077, 0.000, 0.000, 0.000],
        [-0.003, 0.065, 0.723, -0.094, 0.000, -0.023, 0.000],
        [-0.003, 0.083, 0.862, -0.083, 0.021, -0.030, 0.000],
        [-0.003, 0.216, 0.862, -0.066, 0.000, -0.022, 0.000],
        [-0.003, 0.330, 0.914, -0.048, 0.000, -0.012, 0.000],
        [-0.003, 0.457, 0.914, -0.051, -0.007, -0.012, 0.000],
    ])

    ada_horizon = AdaptiveHorizon()
    final_act = ada_horizon.adapt_horizon_dplan_hist(continuous_action, discrete_action)
    print("Final Action:", final_act)
    print("Num actions: ", len(final_act))
