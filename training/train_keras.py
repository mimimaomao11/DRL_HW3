import numpy as np
import random
import tensorflow as tf
from collections import deque
import sys
import os
import matplotlib.pyplot as plt

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorld

print("Fast Keras DQN (NO FREEZE VERSION)")

env = GridWorld(mode="random")

state_size = 4
action_size = 4

# ===== Model =====
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(state_size,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(action_size)
    ])

model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

# Bonus: Learning Rate Scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
# Bonus: Gradient Clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
loss_fn = tf.keras.losses.MeanSquaredError()

buffer = deque(maxlen=5000)

gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

episodes = 1000
batch_size = 32

reward_history = []
loss_history = []

# ===== Training =====
for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(40):

        # action
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            q = model(np.array([state]), training=False)
            action = tf.argmax(q[0]).numpy()

        next_state, reward, done = env.step(action)

        buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # ===== train =====
        if len(buffer) > batch_size:
            batch = random.sample(buffer, batch_size)

            states = np.array([b[0] for b in batch])
            actions = np.array([b[1] for b in batch])
            rewards = np.array([b[2] for b in batch])
            next_states = np.array([b[3] for b in batch])
            dones = np.array([b[4] for b in batch])

            next_q = target_model(next_states, training=False)
            max_next_q = tf.reduce_max(next_q, axis=1)

            targets = rewards + gamma * max_next_q.numpy() * (1 - dones)

            with tf.GradientTape() as tape:
                q_values = model(states, training=True)
                action_q = tf.reduce_sum(
                    q_values * tf.one_hot(actions, action_size),
                    axis=1
                )

                loss = loss_fn(targets, action_q)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_history.append(loss.numpy())

        if done:
            break

    reward_history.append(total_reward)

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if ep % 50 == 0:
        target_model.set_weights(model.get_weights())
        print(f"[FAST Keras] Episode {ep}, Reward: {total_reward:.2f}")

# save
os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
np.save("results/logs/keras_rewards.npy", reward_history)

# plot
plt.figure()
plt.plot(reward_history)
plt.title("Keras DQN - Reward Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("results/plots/reward_keras.png", dpi=100, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(loss_history)
plt.title("Keras DQN - Loss Curve")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.savefig("results/plots/loss_keras.png", dpi=100, bbox_inches='tight')
plt.close()

print("Keras training completed (FAST VERSION)")