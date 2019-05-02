from functools import wraps
from typing import Optional, List, Any

import cv2
import numpy as np

from habitat.config import Config
from habitat.core.vector_env import VectorEnv
from habitat.core.utils import tile_images
from habitat.utils.visualizations.maps import to_grid, draw_agent, colorize_topdown_map, draw_goal
from habitat.utils.visualizations.maps import COORDINATE_MAX, COORDINATE_MIN


class register:
    mapping = {}
    _cache = {}

    def __init__(self, name):
        self.name = name

    def __call__(self, func):

        @wraps(func)
        def _thunk(env, obs, num_for_rec, img_size):
            cache = self._cache.get(self.name, None)
            cache, img = func(env, obs, num_for_rec, img_size, cache)
            self._cache[self.name] = cache
            return img

        self.mapping[self.name] = _thunk
        return _thunk


@register("top_down_map")
def top_down_map(venv, obs, num_for_rec, img_size, cache):

    last_heights, top_down_maps = cache or ([], []) 

    imgs = [None for _ in range(num_for_rec)]
    for idx in range(num_for_rec):
        state = venv.get_state_at(idx)
        pos = state['position']
        goal_pos = venv.get_goal_pos_at(idx)
        angle = state['2d_angle']
        height = pos[1]
        if cache is None:
            last_heights.append(height)
            print('Getting new topdown map')
            top_down_maps.append(venv.get_topdown_map_at(idx))
        else:
            delta = last_heights[idx] - height
            if delta >= 0.01:
                print('Getting new topdown map')
                top_down_maps[idx] = venv.get_topdown_map_at(idx)
            last_heights[idx] = height

        x, y = to_grid(pos[0],
                       pos[2],
                       COORDINATE_MIN,
                       COORDINATE_MAX,
                       top_down_maps[idx].shape)
        x_goal, y_goal = to_grid(goal_pos[0],
                                 goal_pos[1],
                                 COORDINATE_MIN,
                                 COORDINATE_MAX,
                                 top_down_maps[idx].shape)
        range_x = np.where(np.any(top_down_maps[idx], axis=1))[0]
        range_y = np.where(np.any(top_down_maps[idx], axis=0))[0]
        padding = int(np.ceil(top_down_maps[idx].shape[0] / 125))
        range_x = (
            max(range_x[0] - padding, 0),
            min(range_x[-1] + padding + 1, top_down_maps[idx].shape[0]),
        )
        range_y = (
            max(range_y[0] - padding, 0),
            min(range_y[-1] + padding + 1, top_down_maps[idx].shape[1]),
        )
        img = colorize_topdown_map(top_down_maps[idx])
        img = draw_agent(img, (x, y), 0)
        img = draw_goal(img, (x_goal, y_goal))

        imgs[idx] = img[
            range_x[0]: range_x[1], range_y[0]: range_y[1]
        ]
        imgs[idx] = cv2.resize(imgs[idx], img_size)
        imgs[idx] = np.reshape(imgs[idx],
                               img_size + (3,))

    cache = last_heights, top_down_maps
    return cache, imgs


@register("rgb")
def rgb(venv, obs, num_for_rec, img_size, cache):
    imgs = venv.render(mode='rgb_array',
                       num_envs=num_for_rec,
                       tile=False)

    for i, img in enumerate(imgs):
        imgs[i] = cv2.resize(img, img_size)
    return cache, imgs


class Visualizer:

    def __init__(
        self,
        venv: VectorEnv,
        vis_config: Config,
        num_for_rec: Optional[int] = None
    ) -> None:

        self.venv = venv
        self.num_for_rec = vis_config.NUM_FOR_REC
        self.img_size = (vis_config.H, vis_config.W)
        self.vis_config = vis_config

    def get_image(self, obs) -> List[Any]:
        viz = []

        func = register.mapping[self.vis_config.MAIN]
        imgs = func(self.venv, obs, self.num_for_rec, self.img_size)
        viz.append(imgs)

        for method in self.vis_config.LIST:
            func = register.mapping[method]
            imgs = func(self.venv, obs, self.num_for_rec, self.img_size)
            viz.append(imgs)

        outs = []
        dones = self.venv.episode_over
        for i, imgs in enumerate(zip(*viz)):
            imgs = list(imgs)
            if dones[i]:
                imgs[0] = imgs[0] * 0 + [0, 255, 0]
            img = tile_images(imgs)
            outs.append(img)

        return tile_images(outs)
