import os
import glob
import re

patch_code_reward = """plt.plot(reward_history, alpha=0.3, label="Raw")
    window = 20
    if len(reward_history) >= window:
        smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(reward_history)), smoothed, color='blue', label="MA(20)")
    plt.legend()"""

patch_code_loss = """plt.plot(loss_history, alpha=0.3, label="Raw")
    window_loss = 100
    if len(loss_history) >= window_loss:
        smoothed_loss = np.convolve(loss_history, np.ones(window_loss)/window_loss, mode='valid')
        plt.plot(range(window_loss-1, len(loss_history)), smoothed_loss, color='red', label="MA(100)")
    plt.legend()"""

files = glob.glob("training/train_*.py")

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Check if already patched
    if 'alpha=0.3' in content:
        continue
        
    # Replace reward plot
    content = re.sub(r'plt\.plot\(\s*rewards?\s*\)', patch_code_reward.replace('reward_history', 'rewards'), content)
    content = re.sub(r'plt\.plot\(\s*reward_history\s*\)', patch_code_reward, content)
    
    # Replace loss plot
    content = re.sub(r'plt\.plot\(\s*loss_history\s*\)', patch_code_loss, content)
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)

print("Patched all training scripts!")
