
from __future__ import annotations
from gym import spaces
from typing import Any, ClassVar

import numpy as np
import safety_gymnasium
import random
import torch

from omnisafe.envs.classic_control.envs_from_crabs import SafeEnv
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import Box


class KP():

    def __init__(self):
        self.env_id = 'KuhnPoker'
        self.totalgametime = 3
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)        # 独热编码


    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if not None:
            random.seed(seed)
            torch.random.seed()

        first = random.randint(0, 1)
        P1, P2 = torch.randperm(3)[:2] + 1  # 1=J,2=Q,3=K
        self.state1 = torch.tensor([[first, 0, 0, 0, 0, P1]])
        self.state2 = torch.tensor([[1 - first, 0, 0, 0, 0, P2]])
        self.gametime = 0  # game will stop if self.gametime > self.totalgametime

        obs = torch.cat((self.state1, self.state2), dim=0).detach()
        info = {}

        return obs, info

    def step(  # 0=bet,1=pass
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        action1, action2 = action
        self.gametime += 1
        terminated = 0
        reward1 = reward2 = jackpot = 0
        with torch.no_grad():
            if self.gametime <= self.totalgametime:
                if self.state1[0, 0] == 1:  # P2 first
                    if self.gametime == 1:  # P2 bp
                        self.state1[0, 1 + action2] = 1
                        self.state2[0, 3 + action2] = 1
                    elif self.gametime == 2:  # P1 bp
                        self.state1[0, 3 + action1] = 1
                        self.state2[0, 1 + action1] = 1
                        P1p = self.state2[0, 2]
                        P1b = self.state2[0, 1]
                        P2p = self.state1[0, 2]
                        P2b = self.state1[0, 1]
                        if (P1p and P2p) or (P1p and P2b) or (P1b and P2b):
                            terminated = 1
                    else:
                        terminated = 1
                    if terminated:  # P2 bp
                        P1 = self.state1[0, -1]
                        P2 = self.state2[0, -1]
                        P1p = self.state2[0, 2]
                        P1b = self.state2[0, 1]
                        P2p = self.state1[0, 2]
                        P2b = self.state1[0, 1]

                        if P1p and P2p:  # P1 P2 pass
                            jackpot = 1
                        elif (P2b or action2 == 0) and P1b:  # P1 P2 bet/ P2 pass P1 bet,P2 bet again
                            jackpot = 2
                        elif P2p and P1b and action2 == 1:  # P2 pass P1 bet,P2 pass again
                            jackpot = 1
                        elif P2b and P1p:  # P1 pass P2 bet,P2+1
                            jackpot = 1

                        if (P2p and P1b and action2 == 1):
                            reward2 = -jackpot
                            reward1 = jackpot
                        elif P1 < P2 or (P2b and P1p):  # P1 pass P2 bet,P2+1
                            reward2 = jackpot
                            reward1 = -jackpot
                        else:  # P1 pass P2 bet
                            reward2 = -jackpot
                            reward1 = jackpot

                else:  # P1 first
                    if self.gametime == 1:  # P1 bp
                        self.state2[0, 1 + action1] = 1
                        self.state1[0, 3 + action1] = 1
                    elif self.gametime == 2:  # P2 bp
                        self.state2[0, 3 + action2] = 1
                        self.state1[0, 1 + action2] = 1
                        P1p = self.state2[0, 2]
                        P1b = self.state2[0, 1]
                        P2p = self.state1[0, 2]
                        P2b = self.state1[0, 1]
                        if (P1p and P2p) or (P1b and P2p) or (P1b and P2b):
                            terminated = 1
                    else:
                        terminated = 1
                    if terminated:  # P1 bp
                        P1 = self.state1[0, -1]
                        P2 = self.state2[0, -1]
                        P1p = self.state2[0, 2]
                        P1b = self.state2[0, 1]
                        P2p = self.state1[0, 2]
                        P2b = self.state1[0, 1]

                        if P1p and P2p:  # P1 P2 pass
                            jackpot = 1
                        elif (P1b or action1 == 0) and P2b:  # P1 P2 bet/ P2 pass P1 bet,P2 bet again
                            jackpot = 2
                        elif P1p and P2b and action1 == 1:  # P2 pass P1 bet,P2 pass again
                            jackpot = 1
                        elif P1b and P2p:  # P1 pass P2 bet,P2+1
                            jackpot = 1

                        if (P1p and P2b and action1 == 1):
                            reward1 = -jackpot
                            reward2 = jackpot
                        elif P1 > P2 or (P1b and P2p):  # P1 pass P2 bet,P2+1
                            reward1 = jackpot
                            reward2 = -jackpot
                        else:  # P1 pass P2 bet
                            reward1 = -jackpot
                            reward2 = jackpot
            else:  # overtime
                terminated = 1

            obs = torch.cat((self.state1.clone(), self.state2.clone()), dim=0).detach()
            reward = torch.tensor([[reward1], [reward2]]).detach()  # P1 and P2 reward:[[reward1],[reward2]]
        info = {'final_observation': terminated}
        terminated = torch.tensor(terminated).detach()
        truncated = terminated.clone().detach()
        cost = terminated.float()
        return obs, reward, cost, terminated, truncated, info




@env_register
class kuhnpoker(CMDP):
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
        'kuhnpoker'
    ]

    # def __init__(self):
    #     self.action_space = 2  # 独热编码
    #     self.name = 'kuhnpoker'
    #     self.totalgametime = 3  # 总博弈次数
    #     self.observation_space = 6

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str = 'cpu',
        ** kwargs: Any,
    ) -> None:
        if num_envs == 1:
            # set healthy_reward=0.0 for removing the safety constraint in reward
            self._env = KP()#KP(env_id,num_envs,device,** kwargs)

            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        else:
            raise NotImplementedError('Only support num_envs=1 now.')

        self._env_id = self._env.env_id  # 'KuhnPoker'
        self.totalgametime = 3

        self._device = torch.device(device)

        self._num_envs = num_envs

        self._metadata = 0


    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def step(  # 0=bet,1=pass
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        obs, reward, cost, terminated, truncated, info=self._env.step(action)
        return obs, reward, cost, terminated, truncated, info

if __name__ == "__main__":
    env=kuhnpoker()

    while True:

        s=env.reset()
        print('---------------------------')
        print('state',s)
        terminated=False
        while not terminated:
            P1, P2 = map(int, input('0=bet 1=pass >>>').split())
            nes,rew,terminated,_,_=env.step([P1,P2])
            print('nes',nes)
            print('rew',rew,terminated)
