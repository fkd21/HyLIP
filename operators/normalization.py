import numpy as np
from math import log2, floor, ceil
from core.hardware import Hardware


class Normalization:
    """Base class for normalization layers with latency estimation."""
    
    def __init__(self, hardware: Hardware):
        self.hardware = hardware
        
    def forward(self, x):
        """Forward pass of normalization."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def estimate_latency(self, input_shape, data_type="fp16"):
        """Estimate latency using roofline model."""
        raise NotImplementedError("Subclasses must implement estimate_latency method")
    
    def _get_memory_bound_latency(self, input_size, word_size, use_l2=False):
        """Calculate memory-bound latency."""
        # Memory operations: read input, write output
        memory_ops = 2 * input_size * word_size
        
        # Use appropriate bandwidth based on memory hierarchy
        bandwidth = self.hardware.L2_bandwidth if use_l2 else self.hardware.memory_bandwidth["GPU"]
        return memory_ops / bandwidth
    
    def _get_compute_bound_latency(self, input_size, ops_per_element):
        """Calculate compute-bound latency."""
        total_ops = input_size * ops_per_element
        return total_ops / self.hardware.vector_flops
    
    def _gen_L2_tile_size(self, batch_size, seq_len, hidden_dim, word_size):
        """Generate L2 tile sizes for normalization operations."""
        # Calculate L2 cache capacity in elements
        L2_capacity = self.hardware.L2_size // word_size
        
        # For normalization, we need to fit:
        # 1. Input tensor
        # 2. Output tensor
        # 3. Intermediate values
        capacity_per_element = 2.5
        
        # Calculate max elements that fit in L2
        max_elements = L2_capacity // capacity_per_element
        
        # For normalization, must keep the entire hidden dimension intact
        
        # Check if the entire tensor fits in L2
        if batch_size * seq_len * hidden_dim <= max_elements:
            return batch_size, seq_len, hidden_dim
        
        # Tile along batch and seq_len dimensions
        max_batch_seq_pairs = max_elements // hidden_dim
        
        # Prioritize keeping more sequence length in a tile
        if batch_size <= max_batch_seq_pairs:
            # Can fit all batches, tile only on sequence length
            tile_batch = batch_size
            tile_seq = max(1, max_batch_seq_pairs // batch_size)
        else:
            # Need to tile on both batch and sequence dimensions
            tile_seq = 1
            tile_batch = min(batch_size, max(1, max_batch_seq_pairs // tile_seq))
            
            # Try to balance if possible
            if batch_size > 1 and seq_len > 1:
                # Find a more balanced tiling
                for b in range(1, min(batch_size, int(np.sqrt(max_batch_seq_pairs))) + 1):
                    s = max(1, max_batch_seq_pairs // b)
                    if b * s <= max_batch_seq_pairs and b <= batch_size and s <= seq_len:
                        tile_batch = b
                        tile_seq = s
        
        return tile_batch, tile_seq, hidden_dim


class RMSNorm(Normalization):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hardware: Hardware, eps=1e-6):
        super().__init__(hardware)
        self.eps = eps
        
    def forward(self, x, weight=None):
        # Calculate RMS
        variance = np.mean(np.square(x), axis=-1, keepdims=True)
        x_norm = x * np.reciprocal(np.sqrt(variance + self.eps))
        
        if weight is not None:
            x_norm = x_norm * weight
            
        return x_norm
    
    def estimate_latency(self, input_shape, data_type="fp16"):
        if len(input_shape) == 3:
            batch_size, seq_len, hidden_dim = input_shape
        elif len(input_shape) == 2:
            batch_size, hidden_dim = input_shape
            seq_len = 1
        else:
            hidden_dim = input_shape[0]
            batch_size = seq_len = 1
        
        word_size = 2 if data_type == "fp16" else 4
        
        # Get L2 tile sizes
        l2_tile_batch, l2_tile_seq, l2_tile_hidden = self._gen_L2_tile_size(
            batch_size, seq_len, hidden_dim, word_size
        )
        
        # Calculate number of tiles
        batch_tiles = ceil(batch_size / l2_tile_batch)
        seq_tiles = ceil(seq_len / l2_tile_seq)
        
        # RMSNorm operations per element
        ops_per_element = 5
        
        # Additional reduction operations across hidden dimension
        reduction_ops = hidden_dim  # For square and mean reduction
        
        total_latency = 0
        
        # Process each tile
        for b in range(batch_tiles):
            for s in range(seq_tiles):
                # Current tile size
                curr_batch = min(l2_tile_batch, batch_size - b * l2_tile_batch)
                curr_seq = min(l2_tile_seq, seq_len - s * l2_tile_seq)
                
                # Tile size in elements
                tile_size = curr_batch * curr_seq * hidden_dim
                
                # L2 memory operations
                l2_memory_latency = self._get_memory_bound_latency(tile_size, word_size, use_l2=True)
                
                # Element-wise operations
                element_wise_ops = tile_size * ops_per_element
                
                # Reduction operations
                reduction_total_ops = curr_batch * curr_seq * reduction_ops
                
                compute_ops = element_wise_ops + reduction_total_ops
                compute_latency = compute_ops / self.hardware.vector_flops
                
                # L1 memory operations
                l1_memory_latency = self._get_memory_bound_latency(tile_size, word_size, use_l2=False)
                
                # Total latency for this tile
                tile_latency = max(l1_memory_latency, compute_latency) + l2_memory_latency
                total_latency += tile_latency
        
        # Determine bottleneck
        input_size = batch_size * seq_len * hidden_dim
        memory_latency = self._get_memory_bound_latency(input_size, word_size, use_l2=True)
        compute_ops = input_size * ops_per_element + batch_size * seq_len * reduction_ops
        compute_latency = compute_ops / self.hardware.vector_flops
        
        bottleneck = "Memory" if memory_latency > compute_latency else "Compute"
        
        return {
            "latency": total_latency,
            "bottleneck": bottleneck,
            "tiled_latency": total_latency,
            "memory_latency": memory_latency,
            "compute_latency": compute_latency
        }


class LayerNorm(Normalization):
    """Layer Normalization"""
    
    def __init__(self, hardware: Hardware, eps=1e-5):
        super().__init__(hardware)
        self.eps = eps
        
    def forward(self, x, weight=None, bias=None):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean(np.square(x - mean), axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        
        if weight is not None and bias is not None:
            x_norm = x_norm * weight + bias
            
        return x_norm
    
    def estimate_latency(self, input_shape, data_type="fp16"):
        if len(input_shape) == 3:
            batch_size, seq_len, hidden_dim = input_shape
        elif len(input_shape) == 2:
            batch_size, hidden_dim = input_shape
            seq_len = 1
        else:
            hidden_dim = input_shape[0]
            batch_size = seq_len = 1
        
        word_size = 2 if data_type == "fp16" else 4
        
        # Get L2 tile sizes
        l2_tile_batch, l2_tile_seq, l2_tile_hidden = self._gen_L2_tile_size(
            batch_size, seq_len, hidden_dim, word_size
        )
        
        # Calculate number of tiles
        batch_tiles = ceil(batch_size / l2_tile_batch)
        seq_tiles = ceil(seq_len / l2_tile_seq)
        
        # LayerNorm operations per element
        ops_per_element = 8  # subtract, square, divide, multiply, add
        
        # Additional reduction operations across hidden dimension
        # For mean and variance reductions
        reduction_ops = hidden_dim * 2
        
        total_latency = 0
        
        # Process each tile
        for b in range(batch_tiles):
            for s in range(seq_tiles):
                # Current tile size
                curr_batch = min(l2_tile_batch, batch_size - b * l2_tile_batch)
                curr_seq = min(l2_tile_seq, seq_len - s * l2_tile_seq)
                
                # Tile size in elements
                tile_size = curr_batch * curr_seq * hidden_dim
                
                # L2 memory operations
                l2_memory_latency = self._get_memory_bound_latency(tile_size, word_size, use_l2=True)
                
                # Element-wise operations
                element_wise_ops = tile_size * ops_per_element
                
                # Reduction operations
                reduction_total_ops = curr_batch * curr_seq * reduction_ops
                
                compute_ops = element_wise_ops + reduction_total_ops
                compute_latency = compute_ops / self.hardware.vector_flops
                
                # L1 memory operations
                l1_memory_latency = self._get_memory_bound_latency(tile_size, word_size, use_l2=False)
                
                # Total latency for this tile
                tile_latency = max(l1_memory_latency, compute_latency) + l2_memory_latency
                total_latency += tile_latency
        
        # Determine bottleneck
        input_size = batch_size * seq_len * hidden_dim
        memory_latency = self._get_memory_bound_latency(input_size, word_size, use_l2=True)
        compute_ops = input_size * ops_per_element + batch_size * seq_len * reduction_ops
        compute_latency = compute_ops / self.hardware.vector_flops
        
        bottleneck = "Memory" if memory_latency > compute_latency else "Compute"
        
        return {
            "latency": total_latency,
            "bottleneck": bottleneck,
            "tiled_latency": total_latency,
            "memory_latency": memory_latency,
            "compute_latency": compute_latency
        }
