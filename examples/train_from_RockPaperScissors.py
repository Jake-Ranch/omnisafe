# Copyright 2024 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example and template for environment customization."""

from __future__ import annotations

import random
from typing import Any, ClassVar

import torch
from gymnasium import spaces

import omnisafe
from omnisafe.envs.core import CMDP, env_register


# first, define the environment class.
# the most important thing is to add the `env_register` decorator.
@env_register
class CustomExampleEnv(CMDP):

    # define what tasks the environment support.
    _support_envs: ClassVar[list[str]] = ['RockPaperScissor']

    # automatically reset when `terminated` or `truncated`
    need_auto_reset_wrapper = True
    # set `truncated=True` when the total steps exceed the time limit.
    need_time_limit_wrapper = True

    def __init__(self,):
        self.env_id = 'RockPaperScissors'
        self.totalgametime = 200  # 总博弈次数
        self.actionhistory = 6  # 记录双方博弈历史
        self._observation_space = spaces.Box(low=-1,high=1,shape=(self.actionhistory * 7 + 3,),dtype=float)
        self._action_space = spaces.Box(low=-1,high=1,shape=(3,),dtype=float)#spaces.Discrete(3)  # 独热编码
        self.state_dim=self.actionhistory*7+3
        self.action_dim=3
        self._metadata={}


    def step(
        self,
        action_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:  # 0=没出，1=R,2=P,3=S
        action1, action2 = action_
        self.gametime += 1
        terminated = 0
        reward1 = reward2 = 0
        with torch.no_grad():
            if self.gametime < self.totalgametime:
                self.state1[0, :self.actionhistory * 6 -3] = self.state1[0, 3:self.actionhistory * 6].clone()
                self.state2[0, :self.actionhistory * 6 -3] = self.state2[0, 3:self.actionhistory * 6].clone()
                self.state1[0, self.actionhistory * 6:self.actionhistory*7-1] = self.state1[0, self.actionhistory * 6+1:self.actionhistory*7].clone()
                self.state2[0, self.actionhistory * 6:self.actionhistory*7-1] = self.state1[0, self.actionhistory * 6+1:self.actionhistory*7].clone()

                self.state1[0,self.actionhistory*3-3:self.actionhistory*3]=0
                self.state1[0, self.actionhistory * 3 - 3 + action2] = 1
                self.state1[0, self.actionhistory * 6 - 3:self.actionhistory * 6] = 0
                self.state1[0, self.actionhistory * 6 - 3 + action1] = 1

                self.state2[0, self.actionhistory * 3 - 3:self.actionhistory * 3] = 0
                self.state2[0, self.actionhistory * 3 - 3 + action1] = 1
                self.state2[0, self.actionhistory * 6 - 3:self.actionhistory * 6] = 0
                self.state2[0, self.actionhistory * 6 - 3 + action2] = 1

                action1=action1+1
                action2=action2+1
                self.state1[0, -action2] += 1  # 记录对手的出招次数
                self.state2[0, -action1] += 1  # 记录对手的出招次数

                if (action1 == 1 and action2 == 3) or (action1 > action2):
                    # player A win
                    reward1 = 1
                    reward2 = -1
                elif action1 == action2:
                    reward1 = reward2 = 0
                    # 平局
                else:
                    # player B win
                    reward1 = -1
                    reward2 = 1
            else:
                terminated = 1

            self.state1[0, self.actionhistory * 7 - 1] = reward1
            self.state2[0, self.actionhistory * 7 - 1] = reward2

            next_state1 = self.state1.clone()
            next_state2 = self.state2.clone()
            next_state1[0, -3:] = next_state1[0, -3:].clone() / torch.sum(self.state1[0, -3:]).clone()
            next_state2[0, -3:] = next_state2[0, -3:].clone() / torch.sum(self.state2[0, -3:]).clone()

            obs = torch.cat((next_state1, next_state2), dim=0)
            reward = torch.tensor([[reward1], [reward2]])

            info = {'final_observation': obs}
            terminated = torch.tensor(terminated)
            truncated = terminated.clone()
            cost = terminated.float()
        return obs, reward, cost, terminated, truncated, info


    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return 100

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        self.state1 = torch.zeros(1, self.state_dim)  # 十个历史数据，分别是双方的动作和奖惩
        self.state2 = torch.zeros(1, self.state_dim)  # 十个历史数据，分别是双方的动作和奖惩
        self.state1[0, -3:] = 1
        self.state2[0, -3:] = 1
        self.gametime = 0  # 博弈次数
        obs=torch.cat((self.state1, self.state2), dim=0).detach()
        return obs,{}

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def close(self) -> None:
        pass

    def render(self) -> Any:
        pass


# Then you can use it like this:
agent = omnisafe.Agent(
    'QPG',
    # 'RMPG',
    # 'RPG',
    # 'RockPaperScissor',
    'kuhnpoker',
)
agent.learn()
