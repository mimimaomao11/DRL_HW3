import numpy as np
import torch
import sys
import os
import json

# 加入路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn import DQN
from models.dueling_dqn import DuelingDQN
from models.rainbow_dqn import RainbowDQN

def export_all():
    models_to_export = ["dqn", "double", "dueling", "rainbow"]
    all_q_tables = {}
    
    for model_name in models_to_export:
        print(f"Processing {model_name}...")
        if model_name == "dueling":
            model = DuelingDQN()
        elif model_name == "rainbow":
            model = RainbowDQN(num_atoms=51, v_min=-10.0, v_max=10.0)
        else:
            model = DQN()
            
        model_path = f"results/models/{model_name}.pth"
        if not os.path.exists(model_path):
            print(f"Warning: {model_name} model not found. Generating zero Q-table.")
            all_q_tables[model_name] = np.zeros((4,4,4)).tolist()
            continue
            
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
        except Exception as e:
            print(f"Warning: Failed to load {model_name}: {e}. Generating zero Q-table.")
            all_q_tables[model_name] = np.zeros((4,4,4)).tolist()
            continue
        
        q_table = np.zeros((4,4,4))
        support = torch.linspace(-10.0, 10.0, 51)
        
        for i in range(4):
            for j in range(4):
                # Static mode goal is at [0,0]
                state = np.array([i,j,0,0])
                state = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    if model_name == "rainbow":
                        # Set noise to zero for evaluation by not calling reset_noise
                        dist = model(state)
                        q = (dist * support).sum(dim=2).numpy()[0]
                    else:
                        q = model(state).numpy()[0]
                        
                q_table[i,j] = q
                
        all_q_tables[model_name] = q_table.tolist()
        
    os.makedirs("web", exist_ok=True)
    with open("web/q_values.json", "w") as f:
        json.dump(all_q_tables, f)
        
    print("Exported all Q-tables to web/q_values.json")
    
if __name__ == "__main__":
    export_all()