"""Utils for training/fine-tuning scripts."""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK
from prismatic.vla.action_tokenizer import ActionTokenizer


def get_current_action_mask(token_ids):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = cumsum > ACTION_DIM

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )

    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss


def compute_l1_loss_between_discrete_continuous_actions(action_tokenizer, predicted_token_ids, predicted_actions_cont, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    ).reshape(-1, 7).to('cuda')
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, predicted_actions_cont)
    return l1_loss


def tokenize_to_ids(action, action_tokenizer) -> torch.Tensor:
    """
    Tokenize continuous action(s) into discrete token IDs directly.

    Args:
        action: torch.Tensor or np.ndarray of shape (B, T, D) or (B, D)

    Returns:
        token_ids: torch.LongTensor of token IDs with same shape as input[:-1]
    """
    if isinstance(action, torch.Tensor):
        device = action.device
        # Convert to float32 before calling numpy
        action = action.detach().to(torch.float32).cpu().numpy()
    else:
        device = "cpu"

    action = np.clip(action, a_min=action_tokenizer.min_action, a_max=action_tokenizer.max_action)
    discretized_action = np.digitize(action, action_tokenizer.bins-1)
    # token_ids = action_tokenizer.tokenizer.vocab_size - discretized_action

    return torch.LongTensor(discretized_action).to(device)


def compute_alignment_loss(action_tokenizer, gt_tokens, predicted_actions_cont, mask):
    """
    Compute alignment loss between the continuous action predictions and the token logits.

    Args:
        action_tokenizer: ActionTokenizer instance.
        gt_tokens: (B, T, vocab_size) - GT action tokens.
        predicted_actions_cont: (B, T, D) - predicted continuous actions.
        mask: (B, T) - boolean mask to select the valid positions for alignment.

    Returns:
        alignment_loss: scalar torch.Tensor
    """
    # Tokenize the predicted continuous actions
    predicted_action_bins = tokenize_to_ids(predicted_actions_cont, action_tokenizer).reshape(-1).float()

    # Apply mask to both logits and token IDs
    gt_tokens = gt_tokens[mask].float()
    gt_tokens = action_tokenizer.bins - gt_tokens - 1

    # Clip both to be within valid bin range
    # max_bin_idx = action_tokenizer.bin_centers.shape[0] - 1
    # predicted_action_bins = torch.clamp(predicted_action_bins - 1, min=0, max=max_bin_idx)
    # gt_tokens = torch.clamp(gt_tokens - 1, min=0, max=max_bin_idx)

    alignment_loss = torch.nn.functional.l1_loss(gt_tokens, predicted_action_bins)

    correct_preds = (predicted_action_bins == gt_tokens)
    accuracy = torch.mean(correct_preds.sum().float())

    return alignment_loss, accuracy


def compute_alignment_loss_soft_binning(action_tokenizer, gt_token_ids, predicted_actions_cont, mask):
    """
    Differentiable alignment loss using soft binning over action bins.

    Args:
        action_tokenizer: ActionTokenizer instance with bin_centers (NumPy array)
        predicted_actions_cont: Tensor [B, T, D] - predicted continuous actions
        gt_actions_cont: Tensor [B, T, D] - ground-truth continuous actions
        mask: Bool tensor [B, T, D] - optional mask for valid entries

    Returns:
        alignment_loss: scalar torch.Tensor
    """
    # Convert bin centers to tensor
    bin_centers = torch.tensor(
        action_tokenizer.bin_centers,
        device=predicted_actions_cont.device,
        dtype=predicted_actions_cont.dtype
    )  # shape: [BINS]

    # ---- Step 1: Soft binning of predicted actions ---- #
    pred = predicted_actions_cont.unsqueeze(-1)                     # [B, T, D, 1]
    centers = bin_centers.view(1, 1, 1, -1)                         # [1, 1, 1, BINS]
    distances = -torch.abs(pred - centers)                         # [B, T, D, BINS]
    weights = torch.softmax(distances, dim=-1)                     # [B, T, D, BINS]

    bin_ids = torch.arange(bin_centers.shape[0], device=predicted_actions_cont.device).float()
    expected_bin = (weights * bin_ids.view(1, 1, 1, -1)).sum(dim=-1)  # [B, T, D]
    expected_bin = expected_bin.reshape(-1)

    # ---- Step 2: Hard binning of ground-truth actions (no gradients needed) ---- #
    gt_token_ids = gt_token_ids[mask]
    gt_token_ids = gt_token_ids - ACTION_TOKEN_BEGIN_IDX - 1
    gt_actions_clipped = torch.clamp(gt_token_ids, 0, action_tokenizer.bin_centers.shape[0] - 1).float()

    expected_bin_norm = expected_bin / action_tokenizer.n_bins
    gt_actions_norm = gt_actions_clipped / action_tokenizer.n_bins

    alignment_loss = F.l1_loss(expected_bin_norm, gt_actions_norm)

    correct_preds = (expected_bin.round().long() == gt_actions_clipped.long())
    accuracy = correct_preds.float().mean()

    return alignment_loss, accuracy


class EnsembleLoss(nn.Module):
    def __init__(self, action_tokenizer, ensemble_weight=1.0, clamp_weight=0.1, use_soft_clamp=True, 
        device_id='cuda'):
        super().__init__()
        self.tokenizer = action_tokenizer
        self.ensemble_weight = ensemble_weight
        self.clamp_weight = clamp_weight
        self.use_soft_clamp = use_soft_clamp

        # Compute bin width (assumes uniform bins)
        self.bin_width = float(action_tokenizer.bins[1] - action_tokenizer.bins[0])

        self.vocab_size = self.tokenizer.vocab_size
        self.bin_centers = torch.tensor(
            action_tokenizer.bin_centers, dtype=torch.float32, device=device_id
        )

    def forward(self, predicted_cont, predicted_token_ids, ar_logits, gt_actions, gt_token_ids, mask):
        """
        Args:
            predicted_cont: Tensor [B, T, D] - L1 regression head
            predicted_tokens: Tensor [B, T, D] - AR decoded to continuous via bin centers
            ar_logits: Tensor [B, T, V] - raw logits from AR token head
            gt_actions: Tensor [B, T, D] - ground truth continuous actions
            gt_token_ids: LongTensor [B, T, D] - tokenized bin IDs
            mask: BoolTensor [B, T, D] - valid positions
        Returns:
            loss_dict: dict with individual and total losses
        """

        # Compute AR confidence: softmax over vocab
        with torch.no_grad():
            probs = torch.softmax(ar_logits, dim=-1)
            max_probs = probs.max(dim=-1).values[mask]         # [B, T]
            conf = max_probs.view_as(predicted_cont)           # [B, T, D]

        cont_binned_actions = self.decode_token_ids_to_actions_torch(
            predicted_token_ids[mask]
        ).view_as(predicted_cont)

        # Blended actions
        final_actions = conf * cont_binned_actions + (1 - conf) * predicted_cont

        # Masked L1 loss on final blended action
        final_action_loss = F.l1_loss(final_actions, gt_actions)

        # Soft clamping penalty on predicted_cont to stay near bin center
        if self.use_soft_clamp:
            gt_bin_centers = self.decode_token_ids_to_actions_torch(
                gt_token_ids[mask]
            ).view_as(predicted_cont)

            lower = gt_bin_centers - self.bin_width
            upper = gt_bin_centers + self.bin_width

            # soft_excess = torch.relu(torch.abs(predicted_cont - gt_bin_centers) - self.bin_width)
            # soft_clamp_penalty = (soft_excess ** 2).mean()
            soft_clamp_penalty = F.l1_loss(
                predicted_cont, gt_bin_centers)
        else:
            soft_clamp_penalty = torch.tensor(0.0, device=predicted_cont.device)

        # Final weighted loss
        total = (
            self.ensemble_weight * final_action_loss +
            self.clamp_weight * soft_clamp_penalty
        )

        return {
            "ensemble_l1": final_action_loss,
            "soft_clamp_penalty": soft_clamp_penalty,
            "total_ensemble_loss": total,
        }

    def decode_token_ids_to_actions_torch(self, action_token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert action token IDs to continuous action values using bin centers (PyTorch version).

        Args:
            action_token_ids: Tensor [B, T, D] or [N]
            tokenizer_vocab_size: int
            bin_centers: Tensor [num_bins] (same dtype/device as action_token_ids)

        Returns:
            Tensor of continuous actions [same shape as action_token_ids]
        """
        # Recover discretized bin indices from token IDs
        # discretized = self.vocab_size - action_token_ids
        discretized = action_token_ids - ACTION_TOKEN_BEGIN_IDX - 1
        discretized = torch.clamp(discretized - 1, min=0, max=self.bin_centers.shape[0] - 1)

        # Map to continuous bin center values
        return self.bin_centers[discretized]



class DeltaActionDecoder(nn.Module):
    def __init__(self, action_tokenizer, device_id='cuda'):
        super().__init__()
        self.vocab_size = 32000
        self.bin_width = float(action_tokenizer.bins[1] - action_tokenizer.bins[0])
        self.bin_centers = torch.tensor(
            action_tokenizer.bin_centers, dtype=torch.float32, device=device_id
        )

    def forward(self, ar_token_ids, norm_delta, gt_actions, gt_token_ids, mask):
        """
        Args:
            ar_token_ids: LongTensor of shape [B, T, D] (discrete action token IDs)
            ar_hidden_states: FloatTensor of shape [B, T, D] (used for delta prediction)

        Returns:
            final_actions: [B, T, D] continuous actions (bin center + delta)
            delta: [B, T, D] predicted continuous offsets
            bin_centers: [B, T, D] corresponding bin centers
        """
        delta = (self.bin_width / 2) * torch.tanh(norm_delta)

        # ar_bin_centers = self.decode_token_ids_to_actions_torch(
        #     ar_token_ids[mask]).view_as(delta)

        gt_bin_centers = self.decode_token_ids_to_actions_torch(
            gt_token_ids[mask]).view_as(delta)

        # logging
        # comb_actions = ar_bin_centers + delta
        # comb_action_loss = F.l1_loss(comb_actions, gt_actions)

        final_actions = gt_bin_centers + delta
        final_action_loss = F.l1_loss(final_actions, gt_actions)

        return final_action_loss
    
    def decode_token_ids_to_actions_torch(self, action_token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert action token IDs to continuous action values using bin centers (PyTorch version).

        Args:
            action_token_ids: Tensor [B, T, D] or [N]
            tokenizer_vocab_size: int
            bin_centers: Tensor [num_bins] (same dtype/device as action_token_ids)

        Returns:
            Tensor of continuous actions [same shape as action_token_ids]
        """
        # Recover discretized bin indices from token IDs
        # discretized = action_token_ids - ACTION_TOKEN_BEGIN_IDX - 1
        discretized = self.vocab_size - action_token_ids - 1
        discretized = torch.clamp(discretized, min=0, max=self.bin_centers.shape[0] - 1)

        # Map to continuous bin center values
        return self.bin_centers[discretized]