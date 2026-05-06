import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="DQN Dashboard", layout="wide")

st.title("🚀 DQN Variants Dashboard")

st.markdown("Compare different DQN models on GridWorld environment")

# ===== 模型選擇 =====
model_options = ["dqn", "double", "dueling", "keras", "rainbow"]

selected_models = st.multiselect(
    "Select Models to Compare",
    model_options,
    default=["dqn", "double", "dueling"]
)

# ===== smoothing =====
window = st.slider("Smoothing Window", 1, 50, 20)

# ===== 檔案對應 =====
file_map = {
    "dqn": "results/logs/dqn_rewards.txt",
    "double": "results/logs/double_rewards.txt",
    "dueling": "results/logs/dueling_rewards.txt",
    "keras": "results/logs/keras_rewards.npy",
    "rainbow": "results/logs/rainbow_rewards.txt"
}

# ===== 載入資料 =====
def load_data(path):
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".npy"):
            return np.load(path)
        else:
            return np.loadtxt(path)
    except:
        return None

# ===== smoothing function =====
def smooth(y, w):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w)/w, mode='valid')

# ===== 畫圖 =====
fig, ax = plt.subplots(figsize=(10, 6))

loaded_any = False

for model in selected_models:
    data = load_data(file_map[model])

    if data is None:
        st.warning(f"⚠️ {model} data not found")
        continue

    data = smooth(data, window)
    ax.plot(data, label=model.upper())
    loaded_any = True

if loaded_any:
    ax.set_title("DQN Variants Comparison (Smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid()

    st.pyplot(fig)
else:
    st.error("❌ No data available. Please run training first.")

# ===== 單模型細節 =====
st.markdown("---")
st.subheader("📊 Single Model Analysis")

single_model = st.selectbox("Choose Model", model_options)

data = load_data(file_map[single_model])

if data is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.write("Raw Curve")
        fig_raw, ax_raw = plt.subplots()
        ax_raw.plot(data)
        ax_raw.set_title(f"{single_model.upper()} Raw Reward")
        ax_raw.set_xlabel("Episode")
        ax_raw.set_ylabel("Reward")
        st.pyplot(fig_raw)

    with col2:
        st.write("Smoothed Curve")
        fig_s, ax_s = plt.subplots()
        ax_s.plot(smooth(data, window))
        ax_s.set_title(f"{single_model.upper()} Smoothed")
        ax_s.set_xlabel("Episode")
        ax_s.set_ylabel("Reward")
        st.pyplot(fig_s)

    # ===== 統計資訊 =====
    st.markdown("### 📈 Statistics")

    avg_reward = np.mean(data[-50:]) if len(data) >= 50 else np.mean(data)
    max_reward = np.max(data)

    st.write(f"**Final Avg Reward (last 50):** {avg_reward:.2f}")
    st.write(f"**Max Reward:** {max_reward:.2f}")

else:
    st.warning("⚠️ Data not found")

# ===== Footer =====
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")