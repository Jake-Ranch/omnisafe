from __future__ import annotations
import torch
from gym import spaces


from typing import Any, ClassVar

import numpy as np
import safety_gymnasium

from omnisafe.envs.classic_control.envs_from_crabs import SafeEnv
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import Box

class RPS():
    def __init__(self,):
        self.env_id = 'RockPaperScissors'
        self.totalgametime = 200  # 总博弈次数
        self.actionhistory = 6  # 记录双方博弈历史
        self.observation_space = spaces.Box(low=-1,high=1,shape=(self.actionhistory * 7 + 3,),dtype=float)
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,),dtype=float)#spaces.Discrete(3)  # 独热编码
        self.state_dim=self.actionhistory*7+3
        self.action_dim=3
        self.metadata=0

    def reset(self):
        self.state1 = torch.zeros(1, self.state_dim)  # 十个历史数据，分别是双方的动作和奖惩
        self.state2 = torch.zeros(1, self.state_dim)  # 十个历史数据，分别是双方的动作和奖惩
        self.state1[0, -3:] = 1
        self.state2[0, -3:] = 1
        self.gametime = 0  # 博弈次数

        return torch.cat((self.state1, self.state2), dim=0).detach(),{}

    # reward:
    #     R     P      S      A
    # R  0,0   1,-1  -1,1
    # P -1,1   0,0    1,-1
    # S  1,-1 -1,1    0,0

    # B

    def step(self, action_):  # 0=没出，1=R,2=P,3=S
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

            obs = torch.cat((next_state1, next_state2), dim=0)#.detach()
            reward = torch.tensor([[reward1], [reward2]])#.detach()

            info = {'final_observation': terminated}
            terminated = torch.tensor(terminated)#.detach()
            truncated = terminated.clone()#.detach()
            cost = terminated.float()
        return obs, reward, cost, terminated, truncated, info


@env_register
class RockPaperScissors(CMDP):
    need_auto_reset_wrapper = True

    need_time_limit_wrapper = False
    need_action_repeat_wrapper = True

    _support_envs: ClassVar[list[str]] = [
        # 'SafeInvertedPendulum-v2',  # uncomment when pre-trained models is ready.
        # 'SafeInvertedPendulumSwing-v2',
        # 'SafeInvertedPendulumMove-v2',
        # 'MyPendulum-v0',
        # 'MyPendulumTilt-v0',
        # 'MyPendulumUpright-v0',
        'RockPaperScissors'
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str = 'cpu',
        **kwargs: Any,
    ) -> None:
        """Initialize the environment.

        Args:
            env_id (str): Environment id.
            num_envs (int, optional): Number of environments. Defaults to 1.
            device (torch.device, optional): Device to store the data. Defaults to 'cpu'.

        Keyword Args:
            render_mode (str, optional): The render mode, ranging from ``human``, ``rgb_array``, ``rgb_array_list``.
                Defaults to ``rgb_array``.
            camera_name (str, optional): The camera name.
            camera_id (int, optional): The camera id.
            width (int, optional): The width of the rendered image. Defaults to 256.
            height (int, optional): The height of the rendered image. Defaults to 256.
        """

        # super().__init__(env_id)

        if num_envs == 1:
            # set healthy_reward=0.0 for removing the safety constraint in reward
            # self._env = safety_gymnasium.make(id=env_id, autoreset=False, **kwargs)
            self._env = RPS()

            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        else:
            raise NotImplementedError('Only support num_envs=1 now.')
        self._env_id = self._env.env_id
        self._device = torch.device(device)

        self._num_envs = num_envs
        self._metadata = self._env.metadata


    # def __init__(self):
    #     super(RockPaperScissors,self).__init__()

        # self.gamename = 'RockPaperScissors'
        # self.totalgametime = 200  # 总博弈次数
        # self.actionhistory = 5  # 记录双方博弈历史
        # self.observation_space = spaces.Box(low=-1,high=1,shape=(self.actionhistory * 7 + 3,),dtype=float)
        #
        # self.action_space = spaces.Discrete(3)  # 独热编码


    def reset(self):
        obs,info=self._env.reset()

        return obs,info

    # reward:
    #     R     P      S      A
    # R  0,0   1,-1  -1,1
    # P -1,1   0,0    1,-1
    # S  1,-1 -1,1    0,0

    # B

    def step(
        self,
        action: torch.Tensor,# 0=None,1=R,2=P,3=S
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:

        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        return obs, reward, cost, terminated, truncated, info


if __name__ == "__main__":
    env=RockPaperScissors()

    while True:

        s=env.reset()
        print('---------------------------')
        print('state',s)
        terminated=False
        while not terminated:
            P1, P2 = map(int, input('0=Rock 1=Paper 2=Scissors >>>').split())
            nes,rew,terminated,_,_=env.step([P1,P2])
            print('nes',nes)
            print('rew',rew,terminated)

