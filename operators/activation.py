import numpy as np
from math import log2, floor, ceil
from core.hardware import Hardware # Assuming this is the correct path


class Activation:
    """Base class for activation functions with latency estimation."""
    
    def __init__(self, hardware: Hardware):
        self.hardware = hardware
        
    def forward(self, x):
        """Forward pass of activation function."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def estimate_latency(self, input_shape, data_type="fp16"):
        """Estimate latency using roofline model."""
        raise NotImplementedError("Subclasses must implement estimate_latency method")
    
    def _get_memory_bound_latency(self, input_elements: int, word_size: int, use_memory: bool = False, num_rw_tensors: int = 2):
        """Calculate memory-bound latency.
        
        Args:
            input_elements: Number of elements in one tensor involved in the operation.
            word_size: Size of each element in bytes.
            use_memory: Whether to use memory bandwidth (DRAM to L2) vs L2 bandwidth.
            num_rw_tensors: Number of tensor accesses (reads + writes) of size input_elements.
                          Default is 2 (1 read, 1 write). For GLU-like ops (2 reads, 1 write), this would be 3.
        """
        memory_ops_bytes = num_rw_tensors * input_elements * word_size
        
        bandwidth = 0.0
        if use_memory:
            bandwidth = self.hardware.memory_bandwidth.get("GPU", 0) 
        else:
            bandwidth = self.hardware.L2_bandwidth
            
        if bandwidth == 0:
            return float('inf')
        return memory_ops_bytes / bandwidth
    
    def _get_compute_bound_latency(self, input_elements: int, ops_per_element: float) -> float:
        """Calculate compute-bound latency."""
        if self.hardware.vector_flops == 0:
            return float('inf')
        total_ops = input_elements * ops_per_element
        return total_ops / self.hardware.vector_flops
    
    def _estimate_tiled_latency(self, 
                                input_shape_tuple: tuple, 
                                ops_per_element: float, 
                                word_size: int, 
                                use_l2_tiling_path: bool = False, 
                                num_rw_tensors_for_mem: int = 2 
                               ):
        """Estimate latency with consideration for memory hierarchy and tiling.
        
        Args:
            input_shape_tuple: Shape of the primary input tensor this op works on.
            ops_per_element: Number of operations per element of this primary tensor.
            word_size: Size of each element in bytes.
            use_l2_tiling_path: If True, forces the HBM tiling logic if data doesn't fit L2.
                               If False, uses simpler L2-fit or HBM-bulk model.
            num_rw_tensors_for_mem: Number of tensor accesses for memory calculation.
        """
        if not isinstance(input_shape_tuple, tuple):
            return {"latency": float('inf'), "bottleneck": "ShapeError", "memory_latency": float('inf'), "compute_latency": float('inf')}

        input_elements = np.prod(input_shape_tuple)
        if input_elements == 0:
            return {"latency": 0.0, "bottleneck": "N/A", "memory_latency": 0.0, "compute_latency": 0.0}

        l2_capacity_bytes = self.hardware.L2_size
        input_bytes = input_elements * word_size

        # Path A: Simplified model (data fits L2, or bulk HBM transfer if not)
        if not use_l2_tiling_path:
            compute_latency = self._get_compute_bound_latency(input_elements, ops_per_element)
            if input_bytes <= l2_capacity_bytes: # Fits in L2
                memory_latency = self._get_memory_bound_latency(input_elements, word_size, use_memory=False, num_rw_tensors=num_rw_tensors_for_mem) # L2->L1/Reg bandwidth
            else: # Does not fit in L2, must come from HBM
                memory_latency = self._get_memory_bound_latency(input_elements, word_size, use_memory=True, num_rw_tensors=num_rw_tensors_for_mem) # HBM->L2 bandwidth
            
            latency = max(memory_latency, compute_latency)
            bottleneck = "Memory" if memory_latency >= compute_latency else "Compute"
            return {
                "latency": latency, "bottleneck": bottleneck, 
                "memory_latency": memory_latency, "compute_latency": compute_latency,
                "tiled": False
            }

        # Path B: Explicit HBM tiling logic (if use_l2_tiling_path is True AND data doesn't fit L2)
        else:
            l2_capacity_elements = l2_capacity_bytes // word_size
            if input_elements <= l2_capacity_elements:
                # Data fits in L2, fall back to non-tiling path (Path A)
                return self._estimate_tiled_latency(input_shape_tuple, ops_per_element, word_size, 
                                                    use_l2_tiling_path=False, 
                                                    num_rw_tensors_for_mem=num_rw_tensors_for_mem)
            else:
                # Data doesn't fit L2, must tile from HBM.
                tile_buffer_factor = num_rw_tensors_for_mem 
                l2_tile_elements = l2_capacity_elements // tile_buffer_factor
                if l2_tile_elements == 0: l2_tile_elements = 1 

                num_l2_tiles = ceil(input_elements / l2_tile_elements)
                
                # Time to transfer the entire tensor from HBM to L2 (once)
                hbm_total_transfer_latency = self._get_memory_bound_latency(input_elements, word_size, use_memory=True, num_rw_tensors=num_rw_tensors_for_mem)
                
                total_tile_processing_latency = 0
                for i in range(int(num_l2_tiles)):
                    current_tile_elements = min(l2_tile_elements, input_elements - i * l2_tile_elements)
                    if current_tile_elements <= 0: continue

                    tile_compute_latency = self._get_compute_bound_latency(current_tile_elements, ops_per_element)
                    tile_l2_mem_latency = self._get_memory_bound_latency(current_tile_elements, word_size, use_memory=False, num_rw_tensors=num_rw_tensors_for_mem)
                    
                    tile_processing_latency = max(tile_l2_mem_latency, tile_compute_latency)
                    total_tile_processing_latency += tile_processing_latency
                
                # Overall latency is max of total HBM transfer and sum of tile processing times (pipelined model)
                latency = max(hbm_total_transfer_latency, total_tile_processing_latency)
                bottleneck = "Memory (HBM)" if hbm_total_transfer_latency >= total_tile_processing_latency else "Compute/L2_BW"
                
                return {
                    "latency": latency, "bottleneck": bottleneck, "tiled": True,
                    "memory_latency": hbm_total_transfer_latency, 
                    "compute_latency": total_tile_processing_latency 
                }

    def _gen_L2_tile_size(self, batch_size, seq_len, hidden_dim, word_size):
        """Generate L2 tile sizes for operations. (Primarily for Softmax's custom logic)"""
        L2_capacity_bytes = self.hardware.L2_size
        L2_capacity_elements = L2_capacity_bytes // word_size
        
        # Simplified tiling for softmax: process row by row if possible.
        if batch_size * seq_len * hidden_dim * word_size <= L2_capacity_bytes:
            return batch_size, seq_len, hidden_dim # Fits entirely

        tile_max_elements = L2_capacity_elements // 3 
        if tile_max_elements == 0: tile_max_elements = hidden_dim 

        if hidden_dim <= tile_max_elements : 
            num_full_rows_in_tile = tile_max_elements // hidden_dim
            if num_full_rows_in_tile == 0: num_full_rows_in_tile = 1
            l2_tile_batch = 1
            l2_tile_seq = min(seq_len, num_full_rows_in_tile)
            if num_full_rows_in_tile > seq_len: 
                l2_tile_batch = min(batch_size, num_full_rows_in_tile // seq_len)
            return l2_tile_batch, l2_tile_seq, hidden_dim
        else: 
            return 1, 1, hidden_dim # Process one row at a time


class ReLU(Activation):
    ops_per_element = 1 
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
    def forward(self, x): return np.maximum(0, x)
    def estimate_latency(self, input_shape: tuple, data_type="fp16"):
        word_size = 2 if data_type == "fp16" else 4
        return self._estimate_tiled_latency(input_shape, self.ops_per_element, word_size, use_l2_tiling_path=False)

class GELU(Activation):
    ops_per_element = 15 
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
    def forward(self, x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    def estimate_latency(self, input_shape: tuple, data_type="fp16"):
        word_size = 2 if data_type == "fp16" else 4
        return self._estimate_tiled_latency(input_shape, self.ops_per_element, word_size, use_l2_tiling_path=False)

class QuickGELU(Activation):
    ops_per_element = 8 
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
    def forward(self, x): return x * (1 / (1 + np.exp(-1.702 * x)))
    def estimate_latency(self, input_shape: tuple, data_type="fp16"):
        word_size = 2 if data_type == "fp16" else 4
        return self._estimate_tiled_latency(input_shape, self.ops_per_element, word_size, use_l2_tiling_path=False)

class SiLU(Activation):
    ops_per_element = 9 
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
    def forward(self, x): return x * (1 / (1 + np.exp(-x)))
    def estimate_latency(self, input_shape: tuple, data_type="fp16"):
        word_size = 2 if data_type == "fp16" else 4
        return self._estimate_tiled_latency(input_shape, self.ops_per_element, word_size, use_l2_tiling_path=False)

class GLUFamily(Activation): 
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
        self.ops_per_element = 0 # Subclasses must define this

    def estimate_latency(self, input_shape_from_predictor, data_type="fp16"):
        # input_shape_from_predictor is expected to be a list of two identical shapes:
        # [ (B,S,D_branch), (B,S,D_branch) ]
        shape_one_branch = None
        if isinstance(input_shape_from_predictor, list) and len(input_shape_from_predictor) == 2:
            shape_one_branch = input_shape_from_predictor[0]
        elif isinstance(input_shape_from_predictor, tuple): # Fallback if a single tuple is passed
            shape_one_branch = input_shape_from_predictor 
        else:
            return {"latency": float('inf'), "bottleneck": "ShapeError", "memory_latency": float('inf'), "compute_latency": float('inf')}

        if not isinstance(shape_one_branch, tuple):
             return {"latency": float('inf'), "bottleneck": "ShapeError", "memory_latency": float('inf'), "compute_latency": float('inf')}

        word_size = 2 if data_type == "fp16" else 4
        
        # For GLU-like X*Act(Y): Read X, Read Y, Write Z. All same size (shape_one_branch). So 3 tensor movements.
        return self._estimate_tiled_latency(shape_one_branch, self.ops_per_element, word_size, 
                                            use_l2_tiling_path=False, 
                                            num_rw_tensors_for_mem=3)


class GLU(GLUFamily):
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
        self.ops_per_element = 8 + 1 # Sigmoid (~8) + 1 multiply
    def forward(self, x): 
        split_dim = x.shape[-1] // 2
        a, b = np.split(x, [split_dim], axis=-1)
        return a * (1 / (1 + np.exp(-b)))

class GeGLU(GLUFamily):
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
        self.ops_per_element = 15 + 1 # GELU (~15) + 1 multiply
    def forward(self, x):
        split_dim = x.shape[-1] // 2
        a, b = np.split(x, [split_dim], axis=-1)
        return a * (0.5 * b * (1 + np.tanh(np.sqrt(2 / np.pi) * (b + 0.044715 * np.power(b, 3)))))

class SwiGLU(GLUFamily):
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
        self.ops_per_element = 9 + 1 # SiLU (~9) + 1 multiply
    def forward(self, x): 
        split_dim = x.shape[-1] // 2
        a, b = np.split(x, [split_dim], axis=-1)
        return a * (b * (1 / (1 + np.exp(-b))))


class Softmax(Activation):
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)

    def forward(self, x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def estimate_latency(self, input_shape: tuple, data_type="fp16"):
        # Softmax is complex: not purely element-wise due to reduction (sum, max).
        # _estimate_tiled_latency is best for ops where ops_per_element is uniform.
        # For a more accurate Softmax, a dedicated model is better.
        # Using a simplified approach here based on total ops and memory.

        if not isinstance(input_shape, tuple) or not (2 <= len(input_shape) <= 4):
            if isinstance(input_shape, tuple) and input_shape: # Check if input_shape is not empty
                 reduction_dim_size = input_shape[-1]
                 outer_dims_prod = np.prod(input_shape[:-1]) if len(input_shape) > 1 else 1
            else: 
                print("softmax input_shape is not a tuple or is empty")
                return {"latency": float('inf'), "bottleneck": "ShapeError", "memory_latency": float('inf'), "compute_latency": float('inf')}
        else: 
            if len(input_shape) == 4: # B, H, S_q, S_kv (typical attention scores)
                B, H, S_q, S_kv = input_shape
                reduction_dim_size = S_kv
                outer_dims_prod = B * H * S_q
            elif len(input_shape) == 3: # B, S, D (e.g. final layer softmax)
                B, S, D = input_shape
                reduction_dim_size = D
                outer_dims_prod = B * S
            elif len(input_shape) == 2: # S, D (e.g. single instance)
                S, D = input_shape
                reduction_dim_size = D
                outer_dims_prod = S
        input_elements = outer_dims_prod * reduction_dim_size

        word_size = 2 if data_type == "fp16" else 4
        
        exp_cost = 6 # Approximate FLOPs for one exp()
        # Ops per row: (max_comp) + (sub_max) + (exp_ops) + (sum_exp_comp) + (div_ops)
        ops_per_row = (reduction_dim_size - 1) + reduction_dim_size + (reduction_dim_size * exp_cost) + \
                      (reduction_dim_size - 1) + reduction_dim_size
        total_compute_ops = outer_dims_prod * ops_per_row
        num_rw_tensors_softmax = 2 # Read input, Write output (simplified)

        if input_elements == 0: return {"latency": 0.0, "bottleneck": "N/A", "memory_latency": 0.0, "compute_latency": 0.0}
        effective_ops_per_element = total_compute_ops / input_elements
        # Call the simpler path of _estimate_tiled_latency
        result = self._estimate_tiled_latency(input_shape, effective_ops_per_element, word_size, 
                                              use_l2_tiling_path=False, 
                                              num_rw_tensors_for_mem=num_rw_tensors_softmax)
        return result

