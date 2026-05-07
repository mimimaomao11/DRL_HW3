import torch
import json
import os
import sys

# Add parent dir to path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rainbow_dqn import RainbowDQN

def export_rainbow_q_values():
    # Load existing q_values.json
    q_file = "docs/q_values.json"
    with open(q_file, "r") as f:
        q_data = json.load(f)

    # Initialize model
    num_atoms = 51
    v_min = -10.0
    v_max = 10.0
    support = torch.linspace(v_min, v_max, num_atoms)
    
    model = RainbowDQN(num_atoms=num_atoms, v_min=v_min, v_max=v_max)
    model.load_state_dict(torch.load("results/models/rainbow.pth"))
    model.eval()

    # Static mode positions
    goal_x, goal_y = 0, 0
    
    rainbow_q_table = []
    
    # Generate Q-values for all 4x4 positions
    for x in range(4):
        row_q = []
        for y in range(4):
            # State: [player_x, player_y, goal_x, goal_y]
            state = [float(x), float(y), float(goal_x), float(goal_y)]
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                # For inference without noise, we should ideally disable noise or just compute
                dist = model(state_tensor)
                q_expected = (dist * support).sum(dim=2)
                row_q.append(q_expected[0].tolist())
        rainbow_q_table.append(row_q)
        
    # Update JSON
    q_data["rainbow"] = rainbow_q_table
    
    with open(q_file, "w") as f:
        json.dump(q_data, f)
        
    print("Successfully exported Rainbow DQN Q-values to docs/q_values.json")

if __name__ == "__main__":
    export_rainbow_q_values()
