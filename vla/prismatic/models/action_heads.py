"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX

from mamba_ssm import Mamba2  # Using the official Mamba2 block
from prismatic.models.mamba_mixer import MambaDiffusionHead
from prismatic.models.diffusion_utils import mp_sum, mp_cat, mp_silu, MPFourier
from prismatic.models.dis_model import DisModel
from prismatic.models.vae import ConditionalVAE
from prismatic.models.transformer_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )
        # self.model = MambaDiffusionHead(
        #     transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim,
        #     device='cuda', dtype=torch.bfloat16
        # )


    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


def compute_negative_entropy(obs):
    """
    Computes negative entropy for the entire observation embedding.
    Higher value = More predictable state.
    Lower value = More uncertain state.
    """
    # Normalize obs to create a probability-like distribution
    obs_probs = F.softmax(obs, dim=-1)  # Convert embeddings to a probability distribution
    
    # Compute Shannon entropy
    entropy = -torch.sum(obs_probs * torch.log(obs_probs + 1e-8), dim=-1)  # Avoid log(0) issue
    
    # Compute negative entropy (higher = more predictable, lower = more uncertain)
    negative_entropy = -entropy.unsqueeze(-1)  # Negate entropy to match uncertainty intuition
    # shape: [batch_size, chunk_length, 1]
    
    return negative_entropy


class MambaNoisePredictionModel(nn.Module):
    """
    MambaV2-based noise prediction model
    """
    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
        num_blocks=4
    ):  
        super().__init__()
        print("Using MambaNoisePredictionModel")
        print(f"transformer_hidden_dim: {transformer_hidden_dim}")
        print(f"hidden_dim: {hidden_dim}")
        self.layer_norm1 = nn.LayerNorm(transformer_hidden_dim)
        self.fc1 = nn.Linear(transformer_hidden_dim, hidden_dim)
        # self.layer_norm1 = nn.LayerNorm(transformer_hidden_dim+1)
        # self.fc1 = nn.Linear(transformer_hidden_dim+1, hidden_dim)
        self.relu = nn.ReLU()
        self.mamba_blocks = nn.Sequential(
            *[Mamba2(d_model=hidden_dim, d_state=64, d_conv=4, expand=2)
            for _ in range(num_blocks)]
        )
        # self.mid_mamba_blocks = nn.Sequential(
        #     *[Mamba2(d_model=hidden_dim, d_state=64, d_conv=4, expand=2)
        #     for _ in range(num_blocks//3)]
        # )
        # self.final_mamba_blocks = nn.Sequential(
        #     *[Mamba2(d_model=hidden_dim, d_state=64, d_conv=4, expand=2)
        #     for _ in range(num_blocks//3)]
        # )
        # self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        # self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x, cat_state_entropy=False):
        # x: obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        if cat_state_entropy:   # test
            state_entropy = compute_negative_entropy(x)   # shape: (batch_size, input_dim)
            x = torch.cat([x, state_entropy], dim=-1)

        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        # x = self.relu(x)  # shape: (batch_size, hidden_dim)
        x = mp_silu(x)
        identity = x
        x = self.mamba_blocks(x) # shape: (batch_size, hidden_dim)
        # x = torch.cat([x, identity], dim=-1) # shape: (batch_size, hidden_dim + input_dim)
        # x = x + identity
        x = mp_sum(x, identity)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x



class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps=100,
    ):
        super().__init__()
        self.action_dim = action_dim
        # self.noise_predictor = NoisePredictionModel(
        #     transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        # )
        # self.noise_predictor = MambaNoisePredictionModel(
        #     transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        # )
        self.noise_predictor = MambaDiffusionHead(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim,
            device='cuda', dtype=torch.bfloat16
        )
        # self.noise_predictor = DisModel(embed_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, depth=8, channels=action_dim, device='cuda', dtype=torch.bfloat16)
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps, beta_schedule="squaredcos_cap_v2")
        # self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_diffusion_steps)
        self.num_diffusion_steps = num_diffusion_steps
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        # self.time_encoder = MPFourier(hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # test
        # state_entropy = compute_negative_entropy(ground_truth_actions).expand(-1, -1, noise.shape[-1])

        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred


# class L1RegressionActionHead(nn.Module):
#     """VAE Action Head."""
#     def __init__(
#         self,
#         input_dim=4096,
#         hidden_dim=4096,
#         action_dim=7,
#     ):
#         super().__init__()
#         print("Using VAE Action head!")
#         self.action_dim = action_dim
#         # self.model = VAE(x_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, z_dim=action_dim)
#         self.model = ConditionalVAE(input_dim=NUM_ACTIONS_CHUNK*ACTION_DIM, hidden_dim=1024, latent_dim=NUM_ACTIONS_CHUNK*ACTION_DIM, 
#                                     condition_dim=hidden_dim)

#     def predict_action(self, ground_truth_actions, actions_hidden_states):
#         # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
#         # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
#         # ground_truth_actions: ground-truth actions
#         # - shape: (batch_size, chunk_len, action_dim)
#         batch_size = actions_hidden_states.shape[0]
#         device = actions_hidden_states.device
#         rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
#         action, z, mu, logvar = self.model(ground_truth_actions, rearranged_actions_hidden_states)
#         return action, z, mu, logvar


# class L1RegressionActionHead(nn.Module):
#     """Transformer diffusion head."""
#     def __init__(
#         self,
#         input_dim=4096,
#         hidden_dim=4096,
#         action_dim=7,
#     ):
#         super().__init__()
#         print("Using Transformer Diffusion Action head!")
#         self.action_dim = action_dim
        
#         cfg = FlowmatchingActionHeadConfig()
#         cfg.diffusion_model_cfg = \
#                     {"num_attention_heads":16,
#                     "attention_head_dim":256,
#                     "output_dim":hidden_dim,
#                     "num_layers":2,  # Using 2 layers for faster testing
#                     "dropout":0.1,}
#         cfg.input_embedding_dim = hidden_dim
#         cfg.hidden_size = hidden_dim
#         cfg.max_seq_len = hidden_dim

#         self.model = FlowmatchingActionHead(config=cfg)

#     def predict_noise(self, ground_truth_actions, actions_hidden_states, attn_mask):
#         # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
#         # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
#         # ground_truth_actions: ground-truth actions
#         # - shape: (batch_size, chunk_len, action_dim)
#         print("ground_truth_actions.shape: ", ground_truth_actions.shape)
#         print("actions_hidden_states.shape: ", actions_hidden_states.shape)
#         output = self.model(actions_hidden_states,
#                         ground_truth_actions,
#                         None)
        
#         return output
