# TrafficControlRL

TrafficControlRL is a Reinforcement Learning (RL) project focused on intelligent traffic control using policy optimization methods.  
The objective is to train an agent capable of learning optimal control strategies in a simulated traffic environment in order to improve traffic flow, stability, and overall system efficiency.

The project currently implements a Proximal Policy Optimization (PPO) agent interacting with a custom traffic environment.

---

## 🚦 Project Motivation

Urban traffic systems are complex dynamical systems characterized by:

- Non-linear vehicle interactions
- Delays and congestion propagation
- Time-varying inflow rates
- Stability constraints

Traditional rule-based or fixed-timing controllers are often suboptimal.  
Reinforcement Learning enables adaptive, data-driven control policies learned directly from environment interaction.

This project explores the application of modern policy gradient methods to traffic control problems.

---

## 🧠 Core Features

- Custom traffic simulation environment (`envs.py`)
- PPO implementation from scratch (`PPO.py`)
- Actor-Critic neural architecture
- Continuous action space support
- Training and evaluation notebooks
- Model checkpointing (`*.pt`)

---

## 📁 Project Structure
```yaml
├── PPO.py # PPO agent implementation
├── envs.py # Custom traffic environment
├── main.ipynb # Training notebook
├── test_actor.ipynb # Evaluation notebook
├── *.pt # Trained model checkpoints
├── get-pip.py # Auxiliary script
└── .venv/ # Virtual environment (should not be tracked)
```

