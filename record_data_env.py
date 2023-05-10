from __future__ import annotations

from copy import deepcopy
import pickle

import gymnasium as gym
import pygame
from gymnasium import Env
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from plot_data import PlotData


class RecordDataEnv:
    def __init__(
        self, env: Env, seed=None, save_to="data/recorded_data.pickle"
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.data = None
        self.episode_data = None
        self.filename = save_to

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
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        if self.data is None:
            self.data = []
        else:
            self.data.append(deepcopy(self.episode_data))

        self.env.reset(seed=seed)
        self.env.render()
        self.episode_data = []

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.save_data()
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

    def save_data(self):
        with open(self.filename, "wb") as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        img = self.plot_data()
        breakpoint()

    def plot_data(self):
        breakpoint()
        plot = PlotData(self.env.width, self.env.height, self.env.grid.grid)
        img = plot.render(32)

        return img
