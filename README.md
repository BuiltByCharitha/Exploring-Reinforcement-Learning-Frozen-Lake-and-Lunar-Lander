# Reinforcement Learning: Frozen Lake and Lunar Lander with DDQN

This project explores key reinforcement learning (RL) concepts using **Gymnasium** environments and **PyTorch** implementations. It consists of two main parts: value-based Q-learning experiments on the **Frozen Lake** environment and a **Double Deep Q-Network (DDQN)** implementation to solve the **Lunar Lander v2** environment.

## Part 1: Frozen Lake Environment
In this section, a traditional **Q-learning** algorithm is implemented to navigate the Frozen Lake environment. The project investigates how varying **discount factors (γ)** and **learning rates (α)** affect cumulative rewards and convergence. Through multiple experiments, it is observed that:
- Lower γ values make the agent short-sighted, optimizing for immediate rewards.
- Higher γ values promote long-term planning but may slow convergence.
- Learning rate tuning significantly impacts stability and performance—too high causes oscillation, while too low slows learning.

## Part 2: Lunar Lander with DDQN
The second part implements a **Double Deep Q-Network (DDQN)** from scratch using PyTorch, extending the base `dqn.py` structure. The DDQN employs two neural networks—a local Q-network and a target Q-network—to reduce overestimation bias common in standard DQN. The agent is trained on the **Lunar Lander v2** task, and training performance is tracked using reward plots over episodes. The results demonstrate the agent’s ability to successfully learn landing control strategies and achieve stable convergence, effectively solving the environment.
