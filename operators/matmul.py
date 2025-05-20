import math
from math import log2, floor,ceil
import numpy as np
from core.hardware import Hardware
import multiprocessing as mp




def get_matmul_latency(M,N,K,hardware: Hardware,word_size=16,data_type="fp16",ifdouble=True,batch_size=1):
    result1 = single_matmul_latency(M,N,K,hardware,word_size,data_type,ifdouble)
    matmul_latency1 = result1[0] * batch_size if isinstance(result1, tuple) else result1 * batch_size
    
    result2 = single_matmul_latency(M,N,K*batch_size,hardware,word_size,data_type,ifdouble)
    matmul_latency2 = result2[0] if isinstance(result2, tuple) else result2
    matmul_latency2 += (batch_size-1)*M*N*word_size/hardware.memory_bandwidth["GPU"]
    
    min_latency = min(matmul_latency1, matmul_latency2)
    # 将最小延迟和循环顺序作为元组返回
    if isinstance(result1, tuple) and matmul_latency1 <= matmul_latency2:
        return min_latency, result1[1]  # 使用result1的循环顺序
    elif isinstance(result2, tuple):
        return min_latency, result2[1]  # 使用result2的循环顺序
    else:
        return min_latency  # 无循环顺序


def compute_loop_order_latency(loop_order, M, N, K, hardware, word_size, l2_tile_M, l2_tile_N, l2_tile_K, data_type, ifdouble):
    
    if l2_tile_M <= 0 or l2_tile_N <= 0 or l2_tile_K <= 0:
        return 0.0
        
    P_M = math.ceil(M / l2_tile_M)
    P_N = math.ceil(N / l2_tile_N)
    P_K = math.ceil(K / l2_tile_K)

    if P_M == 0 or P_N == 0 or P_K == 0: # No tiles to compute
        return 0.0
        
    N_iter = P_M * P_N * P_K - 1
    if N_iter < 0 : # Handles P_M*P_N*P_K = 0, resulting in N_iter = -1
        N_iter = 0 # No iterations if no tiles or only one tile (0,0,0) for the loop sum

    # --- Calculate base latency components ---
    flops_op = l2_tile_M * l2_tile_N * l2_tile_K * 2
    IO_op = (l2_tile_M * l2_tile_N + l2_tile_M * l2_tile_K + l2_tile_N * l2_tile_K) * word_size

    compute_capability_val = hardware.vector_flops
    if hasattr(hardware, 'tensor_flops_dict') and data_type in hardware.tensor_flops_dict:
        compute_capability_val = hardware.tensor_flops_dict[data_type]
    
    term1_L_comp_tile = float('inf')
    if compute_capability_val > 0:
        term1_L_comp_tile = flops_op / compute_capability_val
        
    term2_L_comp_tile = float('inf')
    if hardware.L2_bandwidth > 0:
        term2_L_comp_tile = IO_op / hardware.L2_bandwidth

    tile_compute_latency = max(term1_L_comp_tile, term2_L_comp_tile)
    
    if hardware.memory_bandwidth["GPU"] <= 0:
        L_read_N_K = float('inf')
        L_read_M_K = float('inf')
        L_read_MK_NK = float('inf')
        L_read_M_N_extra = float('inf')
        L_write_M_N = float('inf')
    else:
        L_read_N_K = (l2_tile_N * l2_tile_K * word_size) / hardware.memory_bandwidth["GPU"]
        L_read_M_K = (l2_tile_M * l2_tile_K * word_size) / hardware.memory_bandwidth["GPU"]
        L_read_MK_NK = ((l2_tile_M * l2_tile_K + l2_tile_N * l2_tile_K) * word_size) / hardware.memory_bandwidth["GPU"]
        L_read_M_N_extra = (l2_tile_M * l2_tile_N * word_size) / hardware.memory_bandwidth["GPU"]
        L_write_M_N = (l2_tile_M * l2_tile_N * word_size) / hardware.memory_bandwidth["GPU"]

    val = [0.0] * 7
    val[1] = L_read_N_K; val[2] = L_read_M_K; val[3] = L_read_MK_NK
    val[4] = L_read_N_K + L_read_M_N_extra
    val[5] = L_read_M_K + L_read_M_N_extra
    val[6] = L_read_MK_NK + L_read_M_N_extra
    
    N_counts = [0] * 7

    def add_to_N_counts(base_read_type_str, has_extra_read, count):
        if count <= 0: return
        if base_read_type_str == "L_read_N_K": N_counts[4 if has_extra_read else 1] += count
        elif base_read_type_str == "L_read_M_K": N_counts[5 if has_extra_read else 2] += count
        elif base_read_type_str == "L_read_MK_NK": N_counts[6 if has_extra_read else 3] += count
            
    if loop_order == "mnk":
        count = P_M * P_N * (P_K - 1); add_to_N_counts("L_read_MK_NK", False, count)
        count = P_M * (P_N - 1)
        if P_K == 1: add_to_N_counts("L_read_N_K", False, count)
        else: add_to_N_counts("L_read_MK_NK", False, count)
        count = P_M - 1
        if P_N == 1 and P_K == 1: add_to_N_counts("L_read_M_K", False, count)
        else: add_to_N_counts("L_read_MK_NK", False, count)
    elif loop_order == "mkn":
        count_n_adv_k_is_0 = P_M * 1 * (P_N - 1); add_to_N_counts("L_read_N_K", False, count_n_adv_k_is_0)
        count_n_adv_k_gt_0 = P_M * (P_K - 1) * (P_N - 1); add_to_N_counts("L_read_N_K", True, count_n_adv_k_gt_0)
        count = P_M * (P_K - 1); add_to_N_counts("L_read_MK_NK", P_N > 1, count)
        count = P_M - 1
        if P_N == 1 and P_K == 1: add_to_N_counts("L_read_M_K", False, count)
        else: add_to_N_counts("L_read_MK_NK", False, count)
    elif loop_order == "nmk":
        count = P_N * P_M * (P_K - 1); add_to_N_counts("L_read_MK_NK", False, count)
        count = P_N * (P_M - 1)
        if P_K == 1: add_to_N_counts("L_read_M_K", False, count)
        else: add_to_N_counts("L_read_MK_NK", False, count)
        count = P_N - 1
        if P_M == 1 and P_K == 1: add_to_N_counts("L_read_N_K", False, count)
        else: add_to_N_counts("L_read_MK_NK", False, count)
    elif loop_order == "nkm":
        count_m_adv_k_is_0 = P_N * 1 * (P_M - 1); add_to_N_counts("L_read_M_K", False, count_m_adv_k_is_0)
        count_m_adv_k_gt_0 = P_N * (P_K - 1) * (P_M - 1); add_to_N_counts("L_read_M_K", True, count_m_adv_k_gt_0)
        count = P_N * (P_K - 1); add_to_N_counts("L_read_MK_NK", P_M > 1, count)
        count = P_N - 1
        if P_M == 1 and P_K == 1: add_to_N_counts("L_read_N_K", False, count)
        else: add_to_N_counts("L_read_MK_NK", False, count)
    elif loop_order == "kmn":
        count_n_adv_k_is_0 = 1 * P_M * (P_N - 1); add_to_N_counts("L_read_N_K", False, count_n_adv_k_is_0)
        count_n_adv_k_gt_0 = (P_K - 1) * P_M * (P_N - 1); add_to_N_counts("L_read_N_K", True, count_n_adv_k_gt_0)
        count_m_adv_k_is_0 = 1 * (P_M - 1)
        count_m_adv_k_gt_0 = (P_K - 1) * (P_M - 1)
        if P_N == 1:
            add_to_N_counts("L_read_M_K", False, count_m_adv_k_is_0)
            add_to_N_counts("L_read_M_K", True, count_m_adv_k_gt_0)
        else:
            add_to_N_counts("L_read_MK_NK", False, count_m_adv_k_is_0)
            add_to_N_counts("L_read_MK_NK", True, count_m_adv_k_gt_0)
        count = P_K - 1; add_to_N_counts("L_read_MK_NK", not (P_M == 1 and P_N == 1), count)
    elif loop_order == "knm":
        count_m_adv_k_is_0 = 1 * P_N * (P_M - 1); add_to_N_counts("L_read_M_K", False, count_m_adv_k_is_0)
        count_m_adv_k_gt_0 = (P_K - 1) * P_N * (P_M - 1); add_to_N_counts("L_read_M_K", True, count_m_adv_k_gt_0)
        count_n_adv_k_is_0 = 1 * (P_N - 1)
        count_n_adv_k_gt_0 = (P_K - 1) * (P_N - 1)
        if P_M == 1:
            add_to_N_counts("L_read_N_K", False, count_n_adv_k_is_0)
            add_to_N_counts("L_read_N_K", True, count_n_adv_k_gt_0)
        else:
            add_to_N_counts("L_read_MK_NK", False, count_n_adv_k_is_0)
            add_to_N_counts("L_read_MK_NK", True, count_n_adv_k_gt_0)
        count = P_K - 1; add_to_N_counts("L_read_MK_NK", not (P_N == 1 and P_M == 1), count)

    Sum_ctrl = 0.0
    Sum_max_ctrl_Lcomp = 0.0
    for i in range(1, 7):
        if N_counts[i] > 0:
            current_val_i = val[i]
            if Sum_ctrl != float('inf'): # Check before adding to prevent inf+finite
                if current_val_i == float('inf'): Sum_ctrl = float('inf')
                else: Sum_ctrl += N_counts[i] * current_val_i
            
            if Sum_max_ctrl_Lcomp != float('inf'): # Check before adding
                if tile_compute_latency == float('inf') or current_val_i == float('inf'):
                    Sum_max_ctrl_Lcomp = float('inf')
                else:
                    Sum_max_ctrl_Lcomp += N_counts[i] * max(current_val_i, tile_compute_latency)
    
    # Corrected Sum_prev_write_lat calculation
    num_writes = 0
    if N_iter >= 0 : # Only calculate if there are iterations to consider
        if loop_order in ["mnk", "nmk"]:
            num_writes = max(0, P_M * P_N - 1)
        elif loop_order == "mkn":
            N_no_write = P_M * (P_K - 1) if P_N == 1 else 0
            num_writes = N_iter - N_no_write
        elif loop_order == "nkm":
            N_no_write = P_N * (P_K - 1) if P_M == 1 else 0
            num_writes = N_iter - N_no_write
        elif loop_order in ["kmn", "knm"]:
            N_no_write = 0 # Write occurs in every relevant iteration step
            num_writes = N_iter - N_no_write
    
    Sum_prev_write_lat = num_writes * L_write_M_N
    if L_write_M_N == float('inf') and num_writes > 0 :
         Sum_prev_write_lat = float('inf')


    TotalLatency_nonpipe = 0.0
    TotalLatency_pipe = 0.0

    effective_total_compute_nonpipe = (P_M * P_N * P_K) * tile_compute_latency
    if tile_compute_latency == float('inf') and (P_M * P_N * P_K > 0):
        effective_total_compute_nonpipe = float('inf')

    if Sum_ctrl == float('inf') or effective_total_compute_nonpipe == float('inf') or Sum_prev_write_lat == float('inf'):
        TotalLatency_nonpipe = float('inf')
    else:
        TotalLatency_nonpipe = Sum_ctrl + effective_total_compute_nonpipe + Sum_prev_write_lat

    final_compute_stage_latency = 0.0
    if P_M * P_N * P_K >= 1:
        if tile_compute_latency == float('inf'): final_compute_stage_latency = float('inf')
        else: final_compute_stage_latency = tile_compute_latency
            
    if Sum_max_ctrl_Lcomp == float('inf') or Sum_prev_write_lat == float('inf') or final_compute_stage_latency == float('inf'):
        TotalLatency_pipe = float('inf')
    else:
        if P_M * P_N * P_K == 1: 
             TotalLatency_pipe = final_compute_stage_latency
        elif N_iter < 0 : # Should be caught by P_M*P_N*P_K == 0 earlier, but as safety
             TotalLatency_pipe = 0.0
        else: 
             TotalLatency_pipe = Sum_max_ctrl_Lcomp + Sum_prev_write_lat + final_compute_stage_latency

    if not ifdouble:    
        return TotalLatency_nonpipe, loop_order
    else:
        return TotalLatency_pipe, loop_order
    #below is the reference logic for matmul latency calculation, for loop is too slow


    # for m, n, k in generate_tile_loops(
    #     ceil(M / l2_tile_M),
    #     ceil(N / l2_tile_N),
    #     ceil(K / l2_tile_K),
    #     loop_order,
    # ):
    #     if m == 0 and n == 0 and k == 0:
    #         continue

    #     # current tile read latency
    #     if m == previous_m and k == previous_k: 
    #         current_tile_read_latency = l2_tile_N * l2_tile_K * word_size/hardware.memory_bandwidth["GPU"]
    #     elif n == previous_n and k == previous_k:
    #         current_tile_read_latency = l2_tile_M * l2_tile_K * word_size/hardware.memory_bandwidth["GPU"]
    #     else:
    #         current_tile_read_latency = (l2_tile_M * l2_tile_K + l2_tile_N * l2_tile_K)*word_size/hardware.memory_bandwidth["GPU"]
    #     if k > 0 and not (m == previous_m and n == previous_n):
    #         current_tile_read_latency += l2_tile_M * l2_tile_N * word_size/hardware.memory_bandwidth["GPU"]

    #     # previous tile compute latency
    #     current_tile_K_reduction_latency=ceil(
    #         l2_tile_M * l2_tile_N * word_size / hardware.vector_flops
    #     ) + 2 * ceil( l2_tile_M * l2_tile_N * word_size/hardware.memory_bandwidth["GPU"])
    
    #     if k > 0:
    #         previous_tile_compute_latency+= previous_tile_K_reduction_latency

    #     # previous tile write latency
    #     if m == previous_m and n == previous_n:
    #         previous_tile_write_latency = 0
    #     else:
    #         previous_tile_write_latency = l2_tile_M * l2_tile_N * word_size / hardware.memory_bandwidth["GPU"]

    #     # read current tile, compute previous tile, write previous tile
    #     if ifdouble:  # pipelined
    #         total_latency += (
    #             max(
    #                 current_tile_read_latency, previous_tile_compute_latency
    #             )
    #             + previous_tile_write_latency
    #         )
    #     else:  # non-pipelined
    #         total_latency += (
    #             current_tile_read_latency
    #             + previous_tile_compute_latency
    #             + previous_tile_write_latency
    #         )

    #     previous_m = m
    #     previous_n = n
    #     previous_k = k
        
        


        

def evaluate_loop_order(loop_order, M, N, K, hardware, word_size, l2_tile_M, l2_tile_N, l2_tile_K, data_type, ifdouble):
    result = compute_loop_order_latency(loop_order, M, N, K, hardware, word_size, 
                                        l2_tile_M, l2_tile_N, l2_tile_K, data_type, ifdouble)
    if isinstance(result, tuple):
        return result
    else:
        return (result, loop_order)

def evaluate_config(args):
    config, M, N, K, hardware, word_size, data_type, ifdouble = args
    l2_tile_M, l2_tile_N, l2_tile_K = config
    loop_orders = ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
    
    results_with_orders = []
    for lo in loop_orders:
        res = compute_loop_order_latency(lo, M, N, K, hardware, word_size, 
                                         l2_tile_M, l2_tile_N, l2_tile_K, data_type, ifdouble)
        if isinstance(res, tuple):
            results_with_orders.append(res)
        else:
            results_with_orders.append((res, lo))
    
    # Find minimum latency and best loop order for this configuration
    if results_with_orders:
        min_lat, best_order = min(results_with_orders, key=lambda x: x[0])
        return (min_lat, best_order, config)
    return (float('inf'), "mnk", config)

def evaluate_combined_config(args):
    loop_order, config, M, N, K, hardware, word_size, data_type, ifdouble = args
    l2_tile_M, l2_tile_N, l2_tile_K = config
    
    result = compute_loop_order_latency(loop_order, M, N, K, hardware, word_size, 
                                       l2_tile_M, l2_tile_N, l2_tile_K, data_type, ifdouble)
    
    if isinstance(result, tuple):
        return result[0], loop_order, config
    else:
        return result, loop_order, config

def single_matmul_latency(M, N, K, hardware: Hardware, word_size=16, data_type="fp16", ifdouble=True,use_multiprocessing=False):
    # Get capacity limit
    if ifdouble:
        capacity = hardware.L2_size // word_size // 2
    else:
        capacity = hardware.L2_size // word_size
    if use_multiprocessing:
        # Heuristic GPU config
        all_configs = []
        for l2_tile_M in [64, 128, 256, 512, 1024, 2048]:
            for l2_tile_N in [l2_tile_M // 2, l2_tile_M, l2_tile_M * 2]:
                if K <= 12288:
                    l2_K_tiling_factor_list = [1, 2, 4, 8]
                else:
                    l2_K_tiling_factor_list = [K // 1024, K // 2048, K // 4096, K // 8192]
                
                for l2_K_tiling_factor in l2_K_tiling_factor_list:
                    l2_tile_K = ceil(K / l2_K_tiling_factor)
                    l2_tile_K = 2 ** floor(log2(l2_tile_K))
                    
                    if l2_tile_K*l2_tile_M+l2_tile_K*l2_tile_N+l2_tile_M*l2_tile_N <= capacity:
                        all_configs.append((l2_tile_M, l2_tile_N, l2_tile_K))
        
        loop_orders = ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        
        # Flatten all combinations of loop orders and tile configurations
        all_combinations = []
        for loop_order in loop_orders:
            for config in all_configs:
                all_combinations.append((loop_order, config, M, N, K, hardware, word_size, data_type, ifdouble))
        
        # Use a single process pool for all evaluations at once
        with mp.Pool(processes=2*mp.cpu_count()) as pool:
            all_results = pool.map(evaluate_combined_config, all_combinations)
        
        if all_results:
            best_result = min(all_results, key=lambda x: x[0])
            return best_result[0], best_result[1]  # Return latency and loop order
    else:
        # Heuristic GPU config
        all_configs = []
        for l2_tile_M in [64, 128, 256, 512, 1024, 2048]:
            for l2_tile_N in [l2_tile_M // 2, l2_tile_M, l2_tile_M * 2]:
                if K <= 12288:
                    l2_K_tiling_factor_list = [1, 2, 4, 8]
                else:
                    l2_K_tiling_factor_list = [K // 1024, K // 2048, K // 4096, K // 8192]
                
                for l2_K_tiling_factor in l2_K_tiling_factor_list:
                    l2_tile_K = ceil(K / l2_K_tiling_factor)
                    l2_tile_K = 2 ** floor(log2(l2_tile_K))
                    
                    if l2_tile_K*l2_tile_M+l2_tile_K*l2_tile_N+l2_tile_M*l2_tile_N <= capacity:
                        all_configs.append((l2_tile_M, l2_tile_N, l2_tile_K))
        
        loop_orders = ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        
        # Evaluate each configuration with each loop order
        results = []
        for config in all_configs:
            for loop_order in loop_orders:
                result = evaluate_config((config, M, N, K, hardware, word_size, data_type, ifdouble))
                results.append(result)
        
        # Find minimum latency and best loop order
        if results:
            min_result = min(results, key=lambda x: x[0])
            return min_result[0], min_result[1]
    
    return float('inf'), "mnk"




# def gen_L2_tile_size(M, N, K, L2_size, word_size, hardware: Hardware, ifdouble=False):
#     if ifdouble:
#         capacity = L2_size // word_size // 2
#     else:
#         capacity = L2_size // word_size

#     best_tile = None
#     best_score = float('inf')

#     ratio = M / N if N > 0 else 1.0

#     for i in range(5, min(11, int(log2(max(1, M))) + 1)):
#         Mt = 2**i
#         Nt_est = Mt / ratio
#         Nt = 2 ** int(log2(max(1, Nt_est)))
#         if Nt > N or Mt > M:
#             continue

#         for Kt in [32, 64, 128]:
#             if Kt > K:
#                 continue

#             total = Mt * Nt + Mt * Kt + Nt * Kt
#             if total > capacity:
#                 continue

#             compute_t = 2 * Mt * Nt * Kt / hardware.vector_flops
#             memory_t = total * word_size / hardware.memory_bandwidth["GPU"]
#             balance = compute_t / memory_t
#             score = abs(log2(balance))

#             if score < best_score:
#                 best_score = score
#                 best_tile = (Mt, Nt, Kt)

#     if best_tile is None:
#         best_tile = (32, 32, 32)
#     l2_tile_M, l2_tile_N, l2_tile_K = best_tile

#     return l2_tile_M, l2_tile_N, l2_tile_K


# def gen_L1_tile_size(l2_tile_M, l2_tile_N, l2_tile_K, L1_size, word_size, hardware: Hardware, ifdouble=False):
#     if ifdouble:
#         capacity = L1_size // word_size // 2
#     else:
#         capacity = L1_size // word_size

#     best_tile = None
#     best_score = float('inf')

#     Try various power-of-2 sizes, but respect L2 tile constraints
#     for Mt in [2**i for i in range(3, int(log2(l2_tile_M)) + 1)]:  # Start from 8
#         for Nt in [2**j for j in range(3, int(log2(l2_tile_N)) + 1)]:
#             for Kt in [2**k for k in range(3, int(log2(l2_tile_K)) + 1)]:
#                 total = Mt * Nt + Mt * Kt + Nt * Kt
#                 if total > capacity:
#                     continue
    
#                 Calculate balance between compute and memory
#                 compute_t = 2 * Mt * Nt * Kt / hardware.vector_flops
#                 memory_t = (Mt * Nt + Mt * Kt + Nt * Kt) * word_size / hardware.memory_bandwidth["GPU"]
#                 balance = compute_t / memory_t      

#                 score = abs(log2(balance))
#                 if score < best_score:
#                     best_score = score
#                     best_tile = (Mt, Nt, Kt)    
#         l1_tile_M, l1_tile_N, l1_tile_K = best_tile

#     return l1_tile_M, l1_tile_N, l1_tile_K


def generate_tile_loops(loop_M: int, loop_N: int, loop_K: int, loop_order: str):
    assert loop_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
    if loop_order == "mnk":
        for m in range(loop_M):
            for n in range(loop_N):
                for k in range(loop_K):
                    yield m, n, k
    elif loop_order == "mkn":
        for m in range(loop_M):
            for k in range(loop_K):
                for n in range(loop_N):
                    yield m, n, k
    elif loop_order == "nmk":
        for n in range(loop_N):
            for m in range(loop_M):
                for k in range(loop_K):
                    yield m, n, k
    elif loop_order == "nkm":
        for n in range(loop_N):
            for k in range(loop_K):
                for m in range(loop_M):
                    yield m, n, k
    elif loop_order == "knm":
        for k in range(loop_K):
            for n in range(loop_N):
                for m in range(loop_M):
                    yield m, n, k
    elif loop_order == "kmn":
        for k in range(loop_K):
            for m in range(loop_M):
                for n in range(loop_N):
                    yield m, n, k