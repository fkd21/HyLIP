# HyLip: Hybrid Latency Inference Predictor for LLMs

This project implements a fine-grained model for predicting LLM inference latency, going beyond the basic Roofline model. The main components include:

- **Transformer Module Modeling**: Models different attention mechanisms, normalizations, and embedding variants
- **Hardware-Aware Operator Modeling**: Uses Roofline-based latency estimation for basic operators (matmul, activation)
- **Scheduling Simulation**: Models both prefill and decode phases separately
- **Multi-GPU Support**: Accounts for communication overhead in tensor parallelism with different topologies (ring, fully connected)

## Project Structure

- `core/`: Core abstractions and interfaces
- `models/`: Transformer and LLM architecture implementations
- `hardware/`: Hardware specifications and capabilities modeling
- `operators/`: Basic computational operators with latency estimations
- `schedulers/`: Task scheduling and execution simulation

## Usage

```python
# Example usage will be provided once implementation is complete
```

<<<<<<< HEAD
=======
## Requirements

- Python 3.8+
- NumPy
- Matplotlib (for visualization) 
>>>>>>> master
