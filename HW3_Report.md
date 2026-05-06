# Homework 3: DQN and its Variants - Understanding Report

## HW3-1: Naive DQN for Static Mode

### Basic DQN Implementation
Deep Q-Network (DQN) combines traditional Q-learning with deep neural networks. In our GridWorld environment, instead of maintaining a Q-table mapping every state-action pair to a value, a neural network is used to approximate the Q-value function: $Q(s, a; \theta) \approx Q^*(s, a)$. The network takes the environment state (e.g., coordinates of the player and goal) as input and outputs the expected Q-value for all possible actions. 

During training, the loss function aims to minimize the Mean Squared Error (MSE) between the current predicted Q-values and the target Q-values. The target Q-value is calculated using the Bellman equation: $R + \gamma \max_{a'} Q(s', a'; \theta^-)$, where $\theta^-$ represents the fixed weights of a separate Target Network that is updated periodically to ensure stability.

### Experience Replay Buffer
To train the neural network effectively, DQN introduces an **Experience Replay Buffer**. In standard Reinforcement Learning, consecutive steps $(S, A, R, S')$ are highly correlated sequentially. Training a neural network on correlated data can cause it to easily forget previous experiences, overfit to the current trajectory, and ultimately diverge.

The Experience Replay Buffer solves this by storing the agent's transition experiences in a queue. During training, a random mini-batch of experiences is sampled from this buffer instead of using the immediate next step. This random sampling breaks the temporal correlation between consecutive data points, significantly stabilizing the training process and improving sample efficiency by allowing the agent to learn from the same experience multiple times.

---

## HW3-2: Enhanced DQN Variants for Player Mode

While Basic DQN is powerful, it suffers from certain structural flaws. We implemented Double DQN and Dueling DQN to address these limitations.

### Double DQN: Mitigating Q-Value Overestimation
**The Flaw in Basic DQN**: Basic DQN uses the same network to both *select* the best next action and to *evaluate* the Q-value of that action. Because it takes the maximum of noisy estimates, it inherently suffers from systematic overestimation of Q-values. This overestimation can lead the agent to converge on sub-optimal policies.

**How Double DQN Improves It**: Double DQN explicitly decouples action selection from action evaluation. It uses the primary online network ($\theta$) to select the best action, and the fixed target network ($\theta^-$) to evaluate its value:

$$Y_t^{DoubleDQN} = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta^-_t)$$

This simple separation significantly reduces the overestimation bias, leading to more stable learning curves and a more accurate representation of the true action values.

### Dueling DQN: Separating Value and Advantage Streams
**The Flaw in Basic DQN**: In many states within an environment, the choice of action doesn't actually matter much (e.g., moving left or right when far from both the goal and the pit). However, standard DQN forces the network to calculate the absolute Q-value for *every* single action individually, which is computationally redundant.

**How Dueling DQN Improves It**: Dueling DQN changes the neural network architecture itself. It splits the final fully connected layers into two separate streams:
1. **State-Value function $V(s)$**: Estimates how fundamentally "good" or "safe" it is to be in a particular state, completely independent of the action taken.
2. **Advantage function $A(s, a)$**: Estimates the relative advantage (or disadvantage) of taking a specific action $a$ in state $s$ compared to other possible actions.

These two streams are then aggregated at the final output layer:
$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)$$

This architectural improvement allows the network to learn the value of states without having to learn the effect of every single action for those states. It leads to much faster convergence and more robust policies, especially in complex environments where many actions lead to identical outcomes.
