import numpy as np
import random

class GridWorld:
    def __init__(self, mode="static"):
        self.size = 4
        self.mode = mode

    def reset(self):
        if self.mode == "static":
            self.player = [0, 3]
            self.goal = [0, 0]
            self.pit = [0, 1]
            self.wall = [1, 1]

        elif self.mode == "random":
            positions = []

            # 隨機生成不重疊位置
            while len(positions) < 4:
                pos = [random.randint(0, self.size - 1),
                       random.randint(0, self.size - 1)]
                if pos not in positions:
                    positions.append(pos)

            self.player, self.goal, self.pit, self.wall = positions

        return self.get_state()

    def get_state(self):
        # 🔥 建議：把 goal 也放進 state（random mode很重要）
        return np.array(self.player + self.goal) / (self.size - 1)

    def step(self, action):
        x, y = self.player

        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1

        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))

        if [x, y] == self.wall:
            x, y = self.player

        self.player = [x, y]

        if self.player == self.goal:
            return self.get_state(), 2.0, True
        elif self.player == self.pit:
            return self.get_state(), -1.0, True
        else:
            return self.get_state(), -0.1, False