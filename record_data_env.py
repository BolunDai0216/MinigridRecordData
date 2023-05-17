from __future__ import annotations

import pickle
from copy import deepcopy
import numpy as np

import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
from gymnasium import Env
from minigrid.core.actions import Actions
from minigrid.core.constants import TILE_PIXELS
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from plot_data import PlotData


class RecordDataEnv:
    def __init__(
        self,
        env: Env,
        seed=None,
        save_to="data/recorded_data.pickle",
        key_map=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.data = None
        self.episode_data = None
        self.filename = save_to
        self.key_map = key_map
        self.rewards = []
        self.counter = 0

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        self.episode_data.append(
            {
                "agent_pos": self.env.agent_pos,
                "agent_dir": self.env.agent_dir,
            }
        )

        if terminated:
            self.data.append(deepcopy(self.episode_data))
            print("terminated!")
            self.rewards.append(reward)
            self.reset(self.seed)
        elif truncated:
            self.data.append(deepcopy(self.episode_data))
            print("truncated!")
            self.rewards.append(reward)
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        if self.data is None:
            self.data = []
        else:
            self.plot_data(self.counter)
            self.counter += 1

        self.env.reset(seed=seed)
        self.env.render()
        self.episode_data = []
        self.episode_data.append(
            {
                "agent_pos": self.env.agent_pos,
                "agent_dir": self.env.agent_dir,
                "mission": self.env.mission,
            }
        )

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.save_data()
            self.env.close()
            self.closed = True
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            self.key_map[0]: Actions.left,
            self.key_map[1]: Actions.right,
            self.key_map[2]: Actions.forward,
            self.key_map[3]: Actions.toggle,
            self.key_map[4]: Actions.pickup,
            self.key_map[5]: Actions.drop,
            self.key_map[6]: Actions.pickup,
            self.key_map[7]: Actions.drop,
            self.key_map[8]: Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

    def save_data(self):
        with open(self.filename, "wb") as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_data(self, counter):
        plot = PlotData(self.env.width, self.env.height, self.env.grid.grid)
        img = plot.render(TILE_PIXELS)

        fig, ax = plt.subplots(1, 1)
        plt.imshow(img)

        _pos = np.array([d["agent_pos"] for d in self.data[-1] if True])
        ax.plot(
            TILE_PIXELS * _pos[:, 0] + int(TILE_PIXELS / 2),
            TILE_PIXELS * _pos[:, 1] + int(TILE_PIXELS / 2),
            color="cornflowerblue",
            alpha=1.0,
            linewidth=int(TILE_PIXELS / 10),
        )

        # Turn off x/y axis numbering/ticks
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        _ = ax.set_xticklabels([])
        _ = ax.set_yticklabels([])

        plt.savefig(
            f"imgs/minigrid_record{counter}.png",
            dpi=200,
            transparent=False,
            bbox_inches="tight",
        )

        return img
