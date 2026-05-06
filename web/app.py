import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

st.title("🚀 DQN Playground")

model = st.selectbox(
    "Choose Model",
    ["dqn", "double", "dueling", "keras", "rainbow"]
)

st.write(f"Selected: {model}")

file_map = {
    "dqn": "results/logs/dqn_rewards.txt",
    "double": "results/logs/double_rewards.txt",
    "dueling": "results/logs/dueling_rewards.txt",
    "keras": "results/logs/keras_rewards.npy",
    "rainbow": "results/logs/rainbow_rewards.txt"
}

file_path = file_map[model]

if os.path.exists(file_path):
    if file_path.endswith(".npy"):
        data = np.load(file_path)
    else:
        data = np.loadtxt(file_path)

    st.success("Data Loaded!")

    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(f"{model.upper()} Reward Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    st.pyplot(fig)

else:
    st.warning("⚠️ Data not found. Please run training first.")