import numpy as np
import json
from enum import Enum


class TopologyType(Enum):
    """网络拓扑类型"""
    RING = "RING"
    FC = "FC"  # Fully Connected


class Hardware:
    def __init__(self, hardware_config_file):
        with open(hardware_config_file, 'r') as f:
            self.hardware_config = json.load(f)

        # 计算资源
        self.L1_size = self.hardware_config['L1_size']
        self.L2_size = self.hardware_config['L2_size']
        
        # 带宽参数
        # L2_bandwidth: L2缓存和L1缓存之间的带宽
        # memory_bandwidth: 主内存和L2缓存之间的带宽
        # 注意: L1带宽通常非常高，不会成为瓶颈，因此我们不考虑它
        self.L2_bandwidth = self.hardware_config['L2_bandwidth']
        
        # 如果配置中有memory_bandwidth，则使用它；否则使用默认值或L2_bandwidth的一小部分

        self.memory_bandwidth = self.hardware_config['memory_bandwidth']
        
        # Ensure memory_bandwidth always has a "GPU" key
        if "GPU" not in self.memory_bandwidth:
            # If SXM4 exists, prefer that (for A100), otherwise use the first key available
            if "SXM4" in self.memory_bandwidth:
                self.memory_bandwidth["GPU"] = self.memory_bandwidth["SXM4"]
            elif "SXM" in self.memory_bandwidth:
                self.memory_bandwidth["GPU"] = self.memory_bandwidth["SXM"]
            elif "PCIe" in self.memory_bandwidth:
                self.memory_bandwidth["GPU"] = self.memory_bandwidth["PCIe"]
            else:
                # Take the first value from the dictionary
                self.memory_bandwidth["GPU"] = list(self.memory_bandwidth.values())[0]

        # Load FLOPS information
        # The vector_flops and tensor_flops are dictionaries with precision keys
        # For simplicity, we extract the fp32 value for vector_flops and fp16 value for tensor_flops
        # These are used in computations where the precision is not specified
        self.vector_flops_dict = self.hardware_config['vector_flops']
        self.tensor_flops_dict = self.hardware_config['tensor_flops']
        
        # Set default values for vector_flops and tensor_flops for easier access
        # Default to fp32 for vector operations and fp16 for tensor operations
        self.vector_flops = self.vector_flops_dict.get("fp32", list(self.vector_flops_dict.values())[0])
        
        if "fp16" in self.tensor_flops_dict:
            self.tensor_flops = self.tensor_flops_dict["fp16"]
        elif "bf16" in self.tensor_flops_dict:
            self.tensor_flops = self.tensor_flops_dict["bf16"]
        else:
            # Fallback to the first value
            self.tensor_flops = list(self.tensor_flops_dict.values())[0]
        
        # 通信资源（如果配置中存在）
        if 'interconnect' in self.hardware_config:
            self.has_interconnect = True
            interconnect = self.hardware_config['interconnect']
            self.device_count = interconnect['device_count']
            self.topology = TopologyType(interconnect['topology'])
            self.link_bandwidth_per_direction = interconnect['link_bandwidth_per_direction']
            self.link_bandwidth_both_direction = interconnect['link_bandwidth_both_direction']
            self.link_latency = interconnect['link_latency']
            self.link_count_per_device = interconnect['link_count_per_device']
            self.internal_link_bandwidth_per_direction = interconnect['internal_link_bandwidth_per_direction']
            self.flit_size = interconnect['flit_size']
            self.header_size = interconnect['header_size']
            self.max_payload_size = interconnect['max_payload_size']
        else:
            self.has_interconnect = False

    def get_hardware_config(self):
        return self.hardware_config
    
    def has_multi_device(self):
        """检查是否有多设备配置"""
        return self.has_interconnect and self.device_count > 1
    
    def get_vector_flops(self, dtype="fp32"):
        """Get vector FLOPS for a specific data type."""
        return self.vector_flops_dict.get(dtype, self.vector_flops)
    
    def get_tensor_flops(self, dtype="fp16"):
        """Get tensor FLOPS for a specific data type."""
        return self.tensor_flops_dict.get(dtype, self.tensor_flops)

