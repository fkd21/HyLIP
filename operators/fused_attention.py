# fused_attention_estimator.py
import numpy as np
from core.hardware import Hardware # Assuming this path is correct

def get_fused_attention_latency(
    batch_size: int,
    num_q_heads: int,    # Total number of query heads
    num_kv_heads: int,   # Total number of key/value heads (can be < num_q_heads for GQA/MQA)
    seq_len_q: int,      # Sequence length for queries
    seq_len_kv: int,     # Sequence length for keys/values (context length)
    head_dim: int,       # Dimension of each attention head
    hardware: Hardware,
    word_size_bits: int = 16, # e.g., 16 for fp16/bf16
    data_type: str = "fp16",
    is_causal: bool = True # Causal masking affects computation slightly, but more so memory patterns for some impl.
                          # For this simplified model, its impact on FLOPs/IO is minor for overall estimate.
) -> float:
    """
    Estimates the latency of a Fused Attention operation using a simplified Roofline model.

    Args:
        batch_size (B): Batch size.
        num_q_heads (H_q): Number of query heads.
        num_kv_heads (H_kv): Number of key/value heads.
        seq_len_q (S_q): Sequence length for queries.
        seq_len_kv (S_kv): Sequence length for keys/values.
        head_dim (D_h): Dimension of each attention head.
        hardware: Hardware capabilities object.
        word_size_bits: Bits per element (e.g., 16 for fp16).
        data_type: Data type string (e.g., "fp16").
        is_causal: Whether causal masking is applied.

    Returns:
        Estimated latency in seconds.
    """
    word_size_bytes = word_size_bits / 8.0

    # --- 1. Calculate Total FLOPs ---
    # QK^T part: (B, H_q, S_q, D_h) x (B, H_kv, S_kv, D_h)^T -> (B, H_q, S_q, S_kv)
    # Each element in the output score matrix requires D_h mul-adds (2*D_h FLOPs).
    # For GQA/MQA, K/V heads are repeated/grouped for Q heads. The computation effectively happens
    # as if H_kv expands to H_q for the matmul, or Q groups map to KV heads.
    # Total QK^T FLOPs: B * H_q * S_q * S_kv * (2 * D_h)
    flops_qk_t = batch_size * num_q_heads * seq_len_q * seq_len_kv * (2 * head_dim)

    # Softmax part: For each element in (B, H_q, S_q, S_kv) score matrix.
    # Includes exp, sum over S_kv, div. Approx ~5-10 FLOPs per score element. Let's use an estimate.
    # For causal, only about half the S_kv elements are involved per S_q row on average.
    effective_softmax_elements = batch_size * num_q_heads * seq_len_q * seq_len_kv
    if is_causal:
        effective_softmax_elements /= 2 # Rough approximation
    flops_softmax = effective_softmax_elements * 8 # Approximate FLOPs for softmax per element

    # SV (Scores @ V) part: (B, H_q, S_q, S_kv) x (B, H_kv, S_kv, D_h) -> (B, H_q, S_q, D_h)
    # Each element in the output O matrix requires S_kv mul-adds (2*S_kv FLOPs).
    # Total SV FLOPs: B * H_q * S_q * D_h * (2 * S_kv)
    flops_sv = batch_size * num_q_heads * seq_len_q * head_dim * (2 * seq_len_kv)
    
    total_flops = flops_qk_t + flops_softmax + flops_sv

    # --- 2. Calculate Total HBM Memory I/O Bytes ---
    # This is the key optimization of Fused Attention: intermediate score matrix is not moved to/from HBM.
    # Read Q: (B, H_q, S_q, D_h)
    bytes_q_read = batch_size * num_q_heads * seq_len_q * head_dim * word_size_bytes
    # Read K: (B, H_kv, S_kv, D_h)
    bytes_k_read = batch_size * num_kv_heads * seq_len_kv * head_dim * word_size_bytes
    # Read V: (B, H_kv, S_kv, D_h)
    bytes_v_read = batch_size * num_kv_heads * seq_len_kv * head_dim * word_size_bytes
    # Write O: (B, H_q, S_q, D_h)
    bytes_o_write = batch_size * num_q_heads * seq_len_q * head_dim * word_size_bytes

    total_hbm_io_bytes = bytes_q_read + bytes_k_read + bytes_v_read + bytes_o_write
    
    if total_hbm_io_bytes == 0: # Avoid division by zero if S_q or S_kv is 0
        return 0.0

    # --- 3. Calculate Arithmetic Intensity (AI) for HBM ---
    arithmetic_intensity = total_flops / total_hbm_io_bytes

    # --- 4. Apply Roofline Model ---
    # Get peak performance from hardware (ensure these attributes exist in your Hardware class)
    peak_compute_flops = hardware.tensor_flops_dict.get(data_type, hardware.vector_flops) # FLOPs/sec
    peak_memory_bw = hardware.memory_bandwidth.get("GPU", 1e12) # Bytes/sec

    if peak_compute_flops == 0 or peak_memory_bw == 0:
        # print("WARN: Peak compute or memory bandwidth is zero in hardware config.")
        return float('inf') # Cannot compute latency

    # Achievable FLOPs based on Roofline
    achievable_tflops = min(peak_compute_flops, arithmetic_intensity * peak_memory_bw)
    
    if achievable_tflops == 0:
        return float('inf') if total_flops > 0 else 0.0

    estimated_latency = total_flops / achievable_tflops
    
    # Kernel launch overheads can be significant for very small computations.
    # This model doesn't include them but could be added as a fixed small value.
    # e.g., estimated_latency += 5e-6 # Add 5us for kernel launch
    
    return estimated_latency

if __name__ == '__main__':
    # Example Usage (requires a dummy Hardware class for standalone testing)
    class DummyHardware:
        def __init__(self):
            self.tensor_flops_dict = {
                "fp16": 312e12, # 312 TFLOPs/s
                "bf16": 312e12
            }
            self.vector_flops = 16e12 # Backup if data_type not in dict
            self.memory_bandwidth = {"GPU": 1.5e12} # 1.5 TB/s

    dummy_hw = DummyHardware()

    # Parameters similar to a 7B model layer
    B, H_q, S_q, D_h = 1, 32, 1024, 128
    H_kv = H_q # MHA example
    S_kv = S_q 
    
    # For GQA (e.g., Qwen2-72B uses H_q=64, H_kv=8, D_h=128)
    # B, H_q, S_q, D_h = 1, 64, 1024, 128
    # H_kv = 8
    # S_kv = S_q


    latency = get_fused_attention_latency(
        batch_size=B,
        num_q_heads=H_q,
        num_kv_heads=H_kv,
        seq_len_q=S_q,
        seq_len_kv=S_kv,
        head_dim=D_h,
        hardware=dummy_hw,
        word_size_bits=16,
        data_type="fp16",
        is_causal=True
    )
    print(f"Estimated Fused Attention Latency (S_q={S_q}, S_kv={S_kv}): {latency * 1000:.4f} ms")

    S_q_decode, S_kv_decode = 1, 1024 # Example decode phase
    latency_decode = get_fused_attention_latency(
        batch_size=B,
        num_q_heads=H_q,
        num_kv_heads=H_kv,
        seq_len_q=S_q_decode,
        seq_len_kv=S_kv_decode, # Context length
        head_dim=D_h,
        hardware=dummy_hw,
        word_size_bits=16,
        data_type="fp16",
        is_causal=True # Causal still applies for the single query token
    )
    print(f"Estimated Fused Attention Latency (S_q={S_q_decode}, S_kv={S_kv_decode}): {latency_decode * 1000:.4f} ms")
