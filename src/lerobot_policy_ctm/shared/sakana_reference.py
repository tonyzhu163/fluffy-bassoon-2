"""Core components extracted from Sakana AI's Continuous Thought Machines.

Source commit: 4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466
Source files: models/modules.py and models/ctm.py.
Definitions are preserved; unrelated image backbones and task utilities are omitted.
Licensed under Apache-2.0; see third_party/sakana_ctm/LICENSE.
"""
import math
import numpy as np
import torch
from torch import nn

class Identity(nn.Module):
    """
    Identity Module.

    Returns the input tensor unchanged. Useful as a placeholder or a no-op layer
    in nn.Sequential containers or conditional network parts.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Squeeze(nn.Module):
    """
    Squeeze Module.

    Removes a specified dimension of size 1 from the input tensor.
    Useful for incorporating tensor dimension squeezing within nn.Sequential.

    Args:
      dim (int): The dimension to squeeze.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class SynapseUNET(nn.Module):
    """
    UNET-style architecture for the Synapse Model (f_theta1 in the paper).

    This module implements the connections between neurons in the CTM's latent
    space. It processes the combined input (previous post-activation state z^t
    and attention output o^t) to produce the pre-activations (a^t) for the
    next internal tick (Eq. 1 in the paper).

    While a simpler Linear or MLP layer can be used, the paper notes
    that this U-Net structure empirically performed better, suggesting benefit
    from more flexible synaptic connections[cite: 79, 80]. This implementation
    uses `depth` points in linspace and creates `depth-1` down/up blocks.

    Args:
      in_dims (int): Number of input dimensions (d_model + d_input).
      out_dims (int): Number of output dimensions (d_model).
      depth (int): Determines structure size; creates `depth-1` down/up blocks.
      minimum_width (int): Smallest channel width at the U-Net bottleneck.
      dropout (float): Dropout rate applied within down/up projections.
    """
    def __init__(self,
                 out_dims,
                 depth,
                 minimum_width=16,
                 dropout=0.0):
        super().__init__()
        self.width_out = out_dims
        self.n_deep = depth # Store depth just for reference if needed

        # Define UNET structure based on depth
        # Creates `depth` width values, leading to `depth-1` blocks
        widths = np.linspace(out_dims, minimum_width, depth)

        # Initial projection layer
        self.first_projection = nn.Sequential(
            nn.LazyLinear(int(widths[0])), # Project to the first width
            nn.LayerNorm(int(widths[0])),
            nn.SiLU()
        )

        # Downward path (encoding layers)
        self.down_projections = nn.ModuleList()
        self.up_projections = nn.ModuleList()
        self.skip_lns = nn.ModuleList()
        num_blocks = len(widths) - 1 # Number of down/up blocks created

        for i in range(num_blocks):
            # Down block: widths[i] -> widths[i+1]
            self.down_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i]), int(widths[i+1])),
                nn.LayerNorm(int(widths[i+1])),
                nn.SiLU()
            ))
            # Up block: widths[i+1] -> widths[i]
            # Note: Up blocks are added in order matching down blocks conceptually,
            # but applied in reverse order in the forward pass.
            self.up_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i+1]), int(widths[i])),
                nn.LayerNorm(int(widths[i])),
                nn.SiLU()
            ))
            # Skip connection LayerNorm operates on width[i]
            self.skip_lns.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        # Initial projection
        out_first = self.first_projection(x)

        # Downward path, storing outputs for skip connections
        outs_down = [out_first]
        for layer in self.down_projections:
            outs_down.append(layer(outs_down[-1]))
        # outs_down contains [level_0, level_1, ..., level_depth-1=bottleneck] outputs

        # Upward path, starting from the bottleneck output
        outs_up = outs_down[-1] # Bottleneck activation
        num_blocks = len(self.up_projections) # Should be depth - 1

        for i in range(num_blocks):
            # Apply up projection in reverse order relative to down blocks
            # up_projection[num_blocks - 1 - i] processes deeper features first
            up_layer_idx = num_blocks - 1 - i
            out_up = self.up_projections[up_layer_idx](outs_up)

            # Get corresponding skip connection from downward path
            # skip_connection index = num_blocks - 1 - i (same as up_layer_idx)
            # This matches the output width of the up_projection[up_layer_idx]
            skip_idx = up_layer_idx
            skip_connection = outs_down[skip_idx]

            # Add skip connection and apply LayerNorm corresponding to this level
            # skip_lns index also corresponds to the level = skip_idx
            outs_up = self.skip_lns[skip_idx](out_up + skip_connection)

        # The final output after all up-projections
        return outs_up

class SuperLinear(nn.Module):
    """
    SuperLinear Layer: Implements Neuron-Level Models (NLMs) for the CTM.

    This layer is the core component enabling Neuron-Level Models (NLMs),
    referred to as g_theta_d in the paper (Eq. 3). It applies N independent
    linear transformations (or small MLPs when used sequentially) to corresponding
    slices of the input tensor along a specified dimension (typically the neuron
    or feature dimension).

    How it works for NLMs:
    - The input `x` is expected to be the pre-activation history for each neuron,
      shaped (batch_size, n_neurons=N, history_length=in_dims).
    - This layer holds unique weights (`w1`) and biases (`b1`) for *each* of the `N` neurons.
      `w1` has shape (in_dims, out_dims, N), `b1` has shape (1, N, out_dims).
    - `torch.einsum('bni,iog->bno', x, self.w1)` performs N independent matrix
      multiplications in parallel (mapping from dim `i` to `o` for each neuron `n`):
        - For each neuron `n` (from 0 to N-1):
        - It takes the neuron's history `x[:, n, :]` (shape B, in_dims).
        - Multiplies it by the neuron's unique weight matrix `self.w1[:, :, n]` (shape in_dims, out_dims).
        - Resulting in `out[:, n, :]` (shape B, out_dims).
    - The unique bias `self.b1[:, n, :]` is added.
    - The result is squeezed on the last dim (if out_dims=1) and scaled by `T`.

    This allows each neuron `d` to process its temporal history `A_d^t` using
    its private parameters `theta_d` to produce the post-activation `z_d^{t+1}`,
    enabling the fine-grained temporal dynamics central to the CTM[cite: 7, 30, 85].
    It's typically used within the `trace_processor` module of the main CTM class.

    Args:
      in_dims (int): Input dimension (typically `memory_length`).
      out_dims (int): Output dimension per neuron.
      N (int): Number of independent linear models (typically `d_model`).
      T (float): Initial value for learnable temperature/scaling factor applied to output.
      do_norm (bool): Apply Layer Normalization to the input history before linear transform.
      dropout (float): Dropout rate applied to the input.
    """
    def __init__(self,
                 in_dims,
                 out_dims,
                 N,
                 T=1.0,
                 do_norm=False,
                 dropout=0):
        super().__init__()
        # N is the number of neurons (d_model), in_dims is the history length (memory_length)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.in_dims = in_dims # Corresponds to memory_length
        # LayerNorm applied across the history dimension for each neuron independently
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm else Identity()
        self.do_norm = do_norm

        # Initialize weights and biases
        # w1 shape: (memory_length, out_dims, d_model)
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        # b1 shape: (1, d_model, out_dims)
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))
        # Learnable temperature/scaler T
        self.register_parameter('T', nn.Parameter(torch.Tensor([T]))) 

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, expected shape (B, N, in_dims)
                              where B=batch, N=d_model, in_dims=memory_length.
        Returns:
            torch.Tensor: Output tensor, shape (B, N) after squeeze(-1).
        """
        # Input shape: (B, D, M) where D=d_model=N neurons in CTM, M=history/memory length
        out = self.dropout(x)
        # LayerNorm across the memory_length dimension (dim=-1)
        out = self.layernorm(out) # Shape remains (B, N, M)

        # Apply N independent linear models using einsum
        # einsum('BDM,MHD->BDH', ...)
        # x: (B=batch size, D=N neurons, one NLM per each of these, M=history/memory length)
        # w1: (M, H=hidden dims if using MLP, otherwise output, D=N neurons, parallel)
        # b1: (1, D=N neurons, H)
        # einsum result: (B, D, H)
        # Applying bias requires matching shapes, b1 is broadcasted.
        out = torch.einsum('BDM,MHD->BDH', out, self.w1) + self.b1

        # Squeeze the output dimension (assumed to be 1 usually) and scale by T
        # This matches the original code's structure exactly.
        out = out.squeeze(-1) / self.T
        return out

class SakanaCoreOperations:
    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
            """
            Computes synchronisation to be used as a vector representation. 

            A neuron has what we call a 'trace', which is a history (time series) that changes with internal
            recurrence. i.e., it gets longer with every internal tick. There are pre-activation traces
            that are used in the NLMs and post-activation traces that, in theory, are used in this method. 

            We define sychronisation between neuron i and j as the dot product between their respective
            time series. Since there can be many internal ticks, this process can be quite compute heavy as it
            involves many dot products that repeat computation at each step.

            Therefore, in practice, we update the synchronisation based on the current post-activations,
            which we call the 'activated state' here. This is possible because the inputs to synchronisation 
            are only updated recurrently at each step, meaning that there is a linear recurrence we can
            leverage. 

            See Appendix TODO of the Technical Report (TODO:LINK) for the maths that enables this method.
            """

            if synch_type == 'action': # Get action parameters
                n_synch = self.n_synch_action
                neuron_indices_left = self.action_neuron_indices_left
                neuron_indices_right = self.action_neuron_indices_right
            elif synch_type == 'out': # Get input parameters
                n_synch = self.n_synch_out
                neuron_indices_left = self.out_neuron_indices_left
                neuron_indices_right = self.out_neuron_indices_right

            if self.neuron_select_type in ('first-last', 'random'):
                # For first-last and random, we compute the pairwise sync between all selected neurons
                if self.neuron_select_type == 'first-last':
                    if synch_type == 'action': # Use last n_synch neurons for action
                        selected_left = selected_right = activated_state[:, -n_synch:]
                    elif synch_type == 'out': # Use first n_synch neurons for out
                        selected_left = selected_right = activated_state[:, :n_synch]
                else: # Use the randomly selected neurons
                    selected_left = activated_state[:, neuron_indices_left]
                    selected_right = activated_state[:, neuron_indices_right]

                # Compute outer product of selected neurons
                outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
                # Resulting matrix is symmetric, so we only need the upper triangle
                i, j = torch.triu_indices(n_synch, n_synch)
                pairwise_product = outer[:, i, j]

            elif self.neuron_select_type == 'random-pairing':
                # For random-pairing, we compute the sync between specific pairs of neurons
                left = activated_state[:, neuron_indices_left]
                right = activated_state[:, neuron_indices_right]
                pairwise_product = left * right
            else:
                raise ValueError("Invalid neuron selection type")



            # Compute synchronisation recurrently
            if decay_alpha is None or decay_beta is None:
                decay_alpha = pairwise_product
                decay_beta = torch.ones_like(pairwise_product)
            else:
                decay_alpha = r * decay_alpha + pairwise_product
                decay_beta = r * decay_beta + 1

            synchronisation = decay_alpha / (torch.sqrt(decay_beta))
            return synchronisation, decay_alpha, decay_beta

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout):
            """
            Neuron level models are one of the core innovations of the CTM. They apply separate MLPs/linears to 
            each neuron.
            NOTE: the name 'SuperLinear' is largely legacy, but its purpose is to apply separate linear layers
                per neuron. It is sort of a 'grouped linear' function, where the group size is equal to 1. 
                One could make the group size bigger and use fewer parameters, but that is future work.

            NOTE: We used GLU() nonlinearities because they worked well in practice. 
            """
            if deep_nlms:
                return nn.Sequential(
                    nn.Sequential(
                        SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model,
                                    do_norm=do_layernorm_nlm, dropout=dropout),
                        nn.GLU(),
                        SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model,
                                    do_norm=do_layernorm_nlm, dropout=dropout),
                        nn.GLU(),
                        Squeeze(-1)
                    )
                )
            else:
                return nn.Sequential(
                    nn.Sequential(
                        SuperLinear(in_dims=memory_length, out_dims=2, N=d_model,
                                    do_norm=do_layernorm_nlm, dropout=dropout),
                        nn.GLU(),
                        Squeeze(-1)
                    )
                )

    def get_synapses(self, synapse_depth, d_model, dropout):
            """
            The synapse model is the recurrent model in the CTM. It's purpose is to share information
            across neurons. If using depth of 1, this is just a simple single layer with nonlinearity and layernomr.
            For deeper synapse models we use a U-NET structure with many skip connections. In practice this performs
            better as it enables multi-level information mixing.

            The intuition with having a deep UNET model for synapses is that the action of synaptic connections is
            not necessarily a linear one, and that approximate a synapose 'update' step in the brain is non trivial. 
            Hence, we set it up so that the CTM can learn some complex internal rule instead of trying to approximate
            it ourselves.
            """
            if synapse_depth == 1:
                return nn.Sequential(
                    nn.Dropout(dropout),
                    nn.LazyLinear(d_model * 2),
                    nn.GLU(),
                    nn.LayerNorm(d_model)
                )
            else:
                return SynapseUNET(d_model, synapse_depth, 16, dropout)

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):
                """
                1. Set the buffers for selecting neurons so that these indices are saved into the model state_dict.
                2. Set the parameters for learnable exponential decay when computing synchronisation between all 
                    neurons.
                """
                assert synch_type in ('out', 'action'), f"Invalid synch_type: {synch_type}"
                left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
                synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
                self.register_buffer(f'{synch_type}_neuron_indices_left', left)
                self.register_buffer(f'{synch_type}_neuron_indices_right', right)
                self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
            """
            Initialize the left and right neuron indices based on the neuron selection type.
            This complexity is owing to legacy experiments, but we retain that these types of
            neuron selections are interesting to experiment with.
            """
            if self.neuron_select_type=='first-last':
                if synch_type == 'out':
                    neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
                elif synch_type == 'action':
                    neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)

            elif self.neuron_select_type=='random':
                neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
                neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))

            elif self.neuron_select_type=='random-pairing':
                assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
                neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
                neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self))))

            device = self.start_activated_state.device
            return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def calculate_synch_representation_size(self, n_synch):
            """
            Calculate the size of the synchronisation representation based on neuron selection type.
            """
            if self.neuron_select_type == 'random-pairing':
                synch_representation_size = n_synch
            elif self.neuron_select_type in ('first-last', 'random'):
                synch_representation_size = (n_synch * (n_synch + 1)) // 2
            else:
                raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
            return synch_representation_size
