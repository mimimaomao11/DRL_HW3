# Deep Reinforcement Learning - DQN Variants (GridWorld)

## Overview

This project implements and compares several Deep Q-Network (DQN) variants in a GridWorld environment.

Detailed Analysis: Please refer to the [HW3 Understanding Report](HW3_Report.md) for an in-depth explanation of the Basic DQN, Experience Replay Buffer, Double DQN, and Dueling DQN.

Implemented models:

- DQN (Baseline)
- Double DQN
- Dueling DQN
- Keras DQN
- Rainbow DQN (Double + Dueling + PER)

## Environment

GridWorld with three modes:

| Mode | Description |
|------|-------------|
| static | fixed positions |
| player | random player |
| random | all elements random |

### Reward:

- Goal: +2.0
- Pit: -1.0
- Step: -0.1

### State:

[player_x, player_y, goal_x, goal_y]

## Models

### 1. DQN

Basic Q-learning with neural network approximation.

### 2. Double DQN

Reduces overestimation using separate action selection and evaluation.

### 3. Dueling DQN

Separates:

- State Value V(s)
- Advantage A(s,a)

### 4. Keras & Lightning DQN (Random Mode with Training Tips)

Re-implemented using TensorFlow/Keras and PyTorch Lightning. To achieve more stable and efficient learning in the difficult `random` mode, the following **Training Tips (Bonus)** were integrated:

- **Gradient Clipping**: Prevents exploding gradients (`clipnorm` in Keras, `clip_grad_norm_` in PyTorch).
- **Learning Rate Scheduling**: Dynamically decays learning rate over time (`ExponentialDecay` in Keras, `StepLR` in PyTorch).
- Batch training

### 5. Rainbow DQN (Simplified)

Includes:

- Double DQN
- Dueling Network
- Prioritized Experience Replay (PER)

## Results

Below shows the smoothed reward curves comparing all the implemented models. Smoothed curves are used to better visualize the convergence trend by filtering out step-by-step stochastic variance.

![All Models Smoothed Comparison](results/plots/all_models_smooth.png)

*Figure: Training rewards over episodes. Rainbow DQN achieves the highest final reward, while Dueling and Double DQN show improved convergence speed and stability over the Baseline DQN.*

## Observations

- DQN converges slowly but stably
- Double & Dueling improve convergence speed
- Keras performs similarly to PyTorch implementation
  Both frameworks were implemented to verify cross-framework consistency and ensure the DQN logic is correct independent of library-specific abstractions.
- Rainbow achieves higher peak performance but shows higher variance

## Key Insight

Rainbow improves performance but introduces instability due to prioritized sampling.

## How to Run

1. Install dependencies
   ```
   pip install -r requirements.txt
   ```

2. Train models
   ```
   python training/train_dqn.py
   python training/train_double.py
   python training/train_dueling.py
   python training/train_keras.py
   python training/train_rainbow.py
   ```

3. Plot comparison
   ```
   python utils/plot_all.py
   ```

## Interactive Visualization (Streamlit)

```
streamlit run app.py
```

## Project Structure

- env/
- models/
- training/
- utils/
- results/
- app.py

## Conclusion

This project demonstrates how different DQN variants improve learning efficiency and stability, with Rainbow achieving the best performance at the cost of higher variance.