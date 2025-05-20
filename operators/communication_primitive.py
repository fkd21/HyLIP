import numpy as np
from math import ceil
from core.hardware import Hardware, TopologyType


class CommunicationPrimitive:
    """通信原语的基类"""
    
    def __init__(self, hardware: Hardware):
        """
        初始化通信原语
        
        Args:
            hardware: 硬件配置对象
        """
        self.hardware = hardware
        if not hardware.has_interconnect:
            raise ValueError("Hardware configuration does not include interconnect parameters")
    
    def estimate_latency(self, data_size_bytes, data_type="fp16"):
        """
        估计通信延迟
        
        Args:
            data_size_bytes: 数据大小（字节）
            data_type: 数据类型，默认为fp16
            
        Returns:
            延迟估计（秒）
        """
        raise NotImplementedError("Subclasses must implement estimate_latency method")


class AllReduce(CommunicationPrimitive):
    """AllReduce通信原语，用于多设备间的梯度聚合"""
    
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
    
    def estimate_latency(self, input_shape, data_type="fp16"):
        """
        估计AllReduce操作的延迟
        
        Args:
            input_shape: 输入张量的形状
            data_type: 数据类型，默认为fp16
            
        Returns:
            延迟估计（秒）和瓶颈信息的字典
        """
        # 计算数据大小（字节）
        word_size = 2 if data_type == "fp16" else 4  # fp16为2字节，fp32为4字节
        data_size = np.prod(input_shape) * word_size
        
        # 从硬件配置中获取参数
        device_count = self.hardware.device_count
        link_bandwidth_per_direction = self.hardware.link_bandwidth_per_direction
        link_bandwidth_both_direction = self.hardware.link_bandwidth_both_direction
        link_latency = self.hardware.link_latency
        flit_size = self.hardware.flit_size
        header_size = self.hardware.header_size
        max_payload_size = self.hardware.max_payload_size
        link_count_per_device = self.hardware.link_count_per_device
        internal_link_bandwidth_per_direction = self.hardware.internal_link_bandwidth_per_direction
        
        # 根据拓扑结构计算延迟
        if self.hardware.topology == TopologyType.FC:  # 全连接拓扑
            # 计算每个设备之间的带宽
            edge_bandwidth_per_direction = (
                link_bandwidth_per_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_bandwidth_both_direction = (
                link_bandwidth_both_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_latency = link_latency
            
            # 每个设备处理的数据大小
            data_size_per_device = data_size / device_count
            
            # 考虑头部开销的有效数据大小
            effective_data_size_per_device = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            
            # 阶段1：环形规约
            latency = (
                edge_latency
                + effective_data_size_per_device / edge_bandwidth_both_direction
            ) * (device_count - 1)
            
            # 阶段2：广播
            latency += effective_data_size_per_device / edge_bandwidth_per_direction
            
            # 内部链路传输延迟
            latency += (
                data_size / internal_link_bandwidth_per_direction
            )
            
        elif self.hardware.topology == TopologyType.RING:  # 环形拓扑
            # 计算边缘带宽
            edge_bandwidth = link_bandwidth_per_direction * link_count_per_device
            edge_latency = link_latency
            
            # 每个设备处理的数据大小
            data_size_per_device = data_size / device_count
            
            # 考虑头部开销的有效数据大小
            effective_data_size_per_device = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            
            # 每次传输的延迟
            per_transmission_latency = effective_data_size_per_device / edge_bandwidth
            
            # 环形AllReduce需要2(n-1)次传输
            latency = (edge_latency + per_transmission_latency) * (
                (device_count - 1) * 2
            )
            
            # 内部链路传输延迟
            latency += (
                data_size / internal_link_bandwidth_per_direction
            )
            
        else:
            raise NotImplementedError(f"Topology {self.hardware.topology} is not supported")
        
        # 返回延迟和相关信息
        return {
            "latency": latency,
            "topology": self.hardware.topology.value,
            "device_count": device_count,
            "data_size_bytes": data_size,
            "data_size_per_device_bytes": data_size_per_device
        }


class AllGather(CommunicationPrimitive):
    """AllGather通信原语，用于收集所有设备上的数据"""
    
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
    
    def estimate_latency(self, input_shape, data_type="fp16"):
        """
        估计AllGather操作的延迟
        
        Args:
            input_shape: 输入张量的形状
            data_type: 数据类型，默认为fp16
            
        Returns:
            延迟估计（秒）和瓶颈信息的字典
        """
        # 计算数据大小（字节）
        word_size = 2 if data_type == "fp16" else 4
        data_size = np.prod(input_shape) * word_size
        
        # 从硬件配置中获取参数
        device_count = self.hardware.device_count
        link_bandwidth_per_direction = self.hardware.link_bandwidth_per_direction
        link_latency = self.hardware.link_latency
        flit_size = self.hardware.flit_size
        header_size = self.hardware.header_size
        max_payload_size = self.hardware.max_payload_size
        link_count_per_device = self.hardware.link_count_per_device
        
        # 根据拓扑结构计算延迟
        if self.hardware.topology == TopologyType.FC:  # 全连接拓扑
            # 计算每个设备之间的带宽
            edge_bandwidth_per_direction = (
                link_bandwidth_per_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_latency = link_latency
            
            # 考虑头部开销的有效数据大小
            effective_data_size = (
                header_size
                + ceil(data_size / max_payload_size) * header_size
                + data_size
            )
            
            # 全连接拓扑中，每个设备直接向其他所有设备发送数据
            latency = (
                edge_latency
                + effective_data_size / edge_bandwidth_per_direction
            )
            
        elif self.hardware.topology == TopologyType.RING:  # 环形拓扑
            # 计算边缘带宽
            edge_bandwidth = link_bandwidth_per_direction * link_count_per_device
            edge_latency = link_latency
            
            # 考虑头部开销的有效数据大小
            effective_data_size = (
                header_size
                + ceil(data_size / max_payload_size) * header_size
                + data_size
            )
            
            # 每次传输的延迟
            per_transmission_latency = effective_data_size / edge_bandwidth
            
            # 环形AllGather需要(n-1)次传输
            latency = (edge_latency + per_transmission_latency) * (device_count - 1)
            
        else:
            raise NotImplementedError(f"Topology {self.hardware.topology} is not supported")
        
        # 返回延迟和相关信息
        return {
            "latency": latency,
            "topology": self.hardware.topology.value,
            "device_count": device_count,
            "data_size_bytes": data_size
        }


class ReduceScatter(CommunicationPrimitive):
    """ReduceScatter通信原语，用于规约并分散数据"""
    
    def __init__(self, hardware: Hardware):
        super().__init__(hardware)
    
    def estimate_latency(self, input_shape, data_type="fp16"):
        """
        估计ReduceScatter操作的延迟
        
        Args:
            input_shape: 输入张量的形状
            data_type: 数据类型，默认为fp16
            
        Returns:
            延迟估计（秒）和瓶颈信息的字典
        """
        # 计算数据大小（字节）
        word_size = 2 if data_type == "fp16" else 4
        data_size = np.prod(input_shape) * word_size
        
        # 从硬件配置中获取参数
        device_count = self.hardware.device_count
        link_bandwidth_per_direction = self.hardware.link_bandwidth_per_direction
        link_latency = self.hardware.link_latency
        flit_size = self.hardware.flit_size
        header_size = self.hardware.header_size
        max_payload_size = self.hardware.max_payload_size
        link_count_per_device = self.hardware.link_count_per_device
        
        # 每个设备处理的数据大小
        data_size_per_device = data_size / device_count
        
        # 根据拓扑结构计算延迟
        if self.hardware.topology == TopologyType.FC:  # 全连接拓扑
            # 计算每个设备之间的带宽
            edge_bandwidth_per_direction = (
                link_bandwidth_per_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_latency = link_latency
            
            # 考虑头部开销的有效数据大小
            effective_data_size = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            
            # 全连接拓扑中，每个设备直接向其他所有设备发送数据
            latency = (
                edge_latency
                + effective_data_size / edge_bandwidth_per_direction
            ) * (device_count - 1)
            
        elif self.hardware.topology == TopologyType.RING:  # 环形拓扑
            # 计算边缘带宽
            edge_bandwidth = link_bandwidth_per_direction * link_count_per_device
            edge_latency = link_latency
            
            # 考虑头部开销的有效数据大小
            effective_data_size = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            
            # 每次传输的延迟
            per_transmission_latency = effective_data_size / edge_bandwidth
            
            # 环形ReduceScatter需要(n-1)次传输
            latency = (edge_latency + per_transmission_latency) * (device_count - 1)
            
        else:
            raise NotImplementedError(f"Topology {self.hardware.topology} is not supported")
        
        # 返回延迟和相关信息
        return {
            "latency": latency,
            "topology": self.hardware.topology.value,
            "device_count": device_count,
            "data_size_bytes": data_size,
            "data_size_per_device_bytes": data_size_per_device
        }