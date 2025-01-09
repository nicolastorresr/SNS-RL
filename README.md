# Social Network Simulation with Reinforcement Learning
This framework simulates social network dynamics using reinforcement learning (RL). Agents make complex decisions such as forming connections, sharing content, and interacting with peers to maximize influence and information spread. The framework leverages Deep Q-Networks (DQN) to enable dynamic and adaptive agent behaviors.

# Features
- Customizable social network environment.
- RL-driven decision-making with DQN agents.
- Comprehensive analysis of network dynamics (e.g., centrality, clustering, topic propagation).
- Interactive visualization of simulation results.

# Requirements
- Python 3.8+
- Install dependencies with `pip install -r requirements.txt`.

# Usage

1. Setup:
```bash
pip install -r requirements.txt
```
2. Run a Simulation:
```bash
python src/training/train.py
```
3. Use `notebooks/demo_simulation.ipynb` for interactive exploration.
4. Test Framework:
```bash
pytest tests/
```



