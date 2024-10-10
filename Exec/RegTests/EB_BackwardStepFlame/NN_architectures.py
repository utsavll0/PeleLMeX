
import jax.numpy as np
from jax import random
from jax.nn import softmax





# Normal MLP
class MLP:
    def __init__(self, layer_sizes, rng_key, activation):
        """Initialize the MLP with layer sizes, a random key, and an activation function."""
        self.rng_key = rng_key
        self.params = self.init_params(layer_sizes)
        self.activation = activation

    def init_params(self, layer_sizes):
        """Initialize weights and biases for all layers."""
        keys = random.split(self.rng_key, len(layer_sizes) - 1)

        def initialize_layer(input_dim, output_dim, key, scale=1e-1):
            """Helper to initialize weights and biases for a single layer."""
            w_key, b_key = random.split(key)
            weights = scale * random.normal(w_key, (input_dim, output_dim))
            biases = scale * random.normal(b_key, (output_dim,))
            return weights, biases
        
        params = [initialize_layer(m, n, k) for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
        return params

    def predict(self, params, inputs):
        """Compute the forward pass."""
        activations = inputs
        for W, b in params[:-1]:
            outputs = np.dot(activations, W) + b
            activations = self.activation(outputs)
        
        # Output layer
        W, b = params[-1]
        final_outputs = np.dot(activations, W) + b
        return final_outputs


# Gating MLP for Mixture of experts gating
class gatingMLP:
    def __init__(self, layer_sizes, rng_key, activation):
        """Initialize the MLP with layer sizes, RNG key, and an activation function."""
        self.rng_key = rng_key
        self.activation = activation
        self.params = self.init_params(layer_sizes)

    def init_params(self, layer_sizes):
        """Initialize all layers to ones."""
        keys = random.split(self.rng_key, len(layer_sizes) - 1)

        def initialize_layer(input_dim, output_dim, _):
            """Initialize weights and biases of a layer to ones.
               So that equal predictions for each DeepONet"""
            weights = np.ones((input_dim, output_dim))
            biases = np.ones((output_dim,))
            return weights, biases
        
        params = [initialize_layer(m, n, k) for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
        return params

    def predict(self, params, inputs):
        """Compute the forward pass, ending with a softmax layer."""
        activations = inputs
        for W, b in params[:-1]:
            outputs = np.dot(activations, W) + b
            activations = self.activation(outputs)

        # Last layer without activation
        W, b = params[-1]
        outputs = np.dot(activations, W) + b
        return softmax(outputs)



class modifiedMLP:
    def __init__(self, layer_sizes, rng_key, activation):
        """Initialize the Modified MLP with layer sizes, a random key, and an activation function.
          Adopted from "S. Wang, Y. Teng, P. Perdikaris, Understanding and mitigating gradient pathologies in physics-informed neural networks" 
          """
        self.rng_key = rng_key
        self.activation = activation
        self.params = self.init_params(layer_sizes)

    def init_params(self, layer_sizes):
        """Initialize weights and biases for all layers and additional parameters."""
        def initialize_layer(input_dim, output_dim, key, scale=1e-1):
            """Helper to initialize weights and biases for a single layer."""
            w_key, b_key = random.split(key)
            weights = scale * random.normal(w_key, (input_dim, output_dim))
            biases = scale * random.normal(b_key, (output_dim,))
            return weights, biases
        
        # Initialize additional layers separately
        mkey1, mkey2 = random.split(self.rng_key, 2)
        u1, b1 = initialize_layer(layer_sizes[0], layer_sizes[1], mkey1)
        u2, b2 = initialize_layer(layer_sizes[0], layer_sizes[1], mkey2)
        
        # Initialize main network layers
        keys = random.split(self.rng_key, len(layer_sizes) - 1)
        params = [initialize_layer(m, n, k) for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
        
        return (params, u1, u2, b1, b2)
    
    def predict(self, params, inputs):
        """Compute the forward pass with modified architecture."""

        params,u1,u2,b1,b2 = params
        
        U = self.activation(np.dot(inputs, u1) + b1)
        V = self.activation(np.dot(inputs, u2) + b2)
        
        activations = inputs
        for W, b in params[:-1]:
            activations = self.activation(np.dot(activations, W) + b)
            activations = np.multiply(activations, U) + np.multiply(1 - activations, V)
        
        # Output layer
        W, b = params[-1]
        final_outputs = np.dot(activations, W) + b
        return final_outputs






class DeepONet:
    def __init__(self, branch_layers, trunk_layers, activation, rng_key, param_layers=None):
        """
        Initializes the Deep Operator Network (DeepONet) with branch and trunk networks.
        
        Parameters:
        - branch_layers: List of integers defining the sizes of the layers in the branch network.
        - trunk_layers: List of integers defining the sizes of the layers in the trunk network.
        - activation: activation function for non-linearity
        - rng_key: rng_key for further random key generation.
        - param_layers: Optional list of integers defining the sizes of the layers in the parameter network.
        """
        self.rng_key = rng_key
        b_key, t_key, p_key = random.split(self.rng_key, 3)
        # Instantiation of branch and trunk networks with modified MLP architecture
        self.branch_net = modifiedMLP(branch_layers, b_key, activation)
        self.trunk_net = modifiedMLP(trunk_layers, t_key, activation)
        self.num_features = branch_layers[0]
        
        # Optional parameter network
        if param_layers:
            self.param_net = modifiedMLP(param_layers, p_key, activation)
            self.has_param_net = True
            print(f"Model has Param Net\n")
        else:
            self.param_net = None
            self.has_param_net = False
            print(f"Model does not have Param Net\n")
        
        # Collect parameters from both networks
        self.params = (self.branch_net.params, self.trunk_net.params, self.param_net.params if self.param_net else None)
        
    def predict(self, params, u, y, e=None):
        """
        Performs the forward pass through the DeepONet.
        
        Parameters:
        - params: Tuple containing parameters for both the branch and trunk networks.
        - u: Input to the branch network.
        - y: Input to the trunk network.
        - e: Optional input to the parameter network.
        
        Returns:
        - The output of the DeepONet combining the outputs from the branch, trunk, and optionally parameter networks.
        """
        branch_params, trunk_params, param_params = params
        
        B = self.branch_net.predict(branch_params, u)  # Output from the branch network
        T = self.trunk_net.predict(trunk_params, y)  # Output from the trunk network
        
        # Element-wise multiplication of branch and trunk outputs
        G = B * T

        # Apply the optional parameter network if it exists
        if self.has_param_net and e is not None:
            E = self.param_net.predict(param_params, e)
            G *= E
        
        # Summing across the first dimension assuming `branch_layers[0]` represents the output size of the branch net
        g_out = np.sum(G.reshape((-1, self.num_features)), axis=0)
        
        return g_out
