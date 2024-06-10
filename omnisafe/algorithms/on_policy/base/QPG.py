
from __future__ import annotations


from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import gym
import random
import matplotlib.pyplot as plt
import time


from ....envs.kuhnpoker import kuhnpoker
from ....envs.RockPaperScissors import RockPaperScissors

envdict={'kuhnpoker':kuhnpoker(),'RockPaperScissors':RockPaperScissors()}


class Actor(nn.Module):
    '''
    演员Actor网络
    '''

    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, action_dim)
        self.fc3 = nn.Linear(300, 300)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        x = self.ln(F.relu(self.fc3(x)))

        # print(self.fc2(x))

        # out = self.ln(F.relu(self.fc2(x)))

        out = F.softmax(self.fc2(x), dim=-1)
        print(s,out)

        return out
class Critic(nn.Module):
    '''
    评论家Critic网络
    '''

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 1)
        self.fc3 = nn.Linear(300, 300)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        x = self.ln(F.relu(self.fc3(x)))
        out = self.fc2(x)

        return out

@registry.register
class QPG(PolicyGradient):
    def _init_env(self) -> None:
        self._env_id = 'kuhnpoker'

        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        # self._env: OnPolicyAdapter = OnPolicyAdapter(
        #     self._env_id,
        #     self._cfgs.train_cfgs.vector_env_nums,
        #     self._seed,
        #     self._cfgs,
        # )
        # assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
        #     distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        # ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        # self._steps_per_epoch: int = (
        #     self._cfgs.algo_cfgs.steps_per_epoch
        #     // distributed.world_size()
        #     // self._cfgs.train_cfgs.vector_env_nums
        # )
        # self._env=RockPaperScissor()


    def __init__(self):
        self.gamma = 1#0.99
        self.lr_a = 3e-5
        self.lr_c = 5e-5

        self.env = envdict[self._env_id]

        self.action_dim = self.env.action_space  # 获取描述行动的数据维度
        self.state_dim = self.env.observation_space  # 获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)  # 创建演员网络
        self.critic = Critic(self.state_dim)  # 创建评论家网络

        self.actor2 = Actor(self.action_dim, self.state_dim)  # 创建演员网络
        self.critic2 = Critic(self.state_dim)  # 创建评论家网络

        self.actor_optim = torch.optim.SGD(self.actor.parameters(),lr=self.lr_a)
        self.critic_optim = torch.optim.SGD(self.critic.parameters(),lr=self.lr_c)

        self.actor_optim2 = torch.optim.SGD(self.actor2.parameters(),lr=self.lr_a)
        self.critic_optim2 = torch.optim.SGD(self.critic2.parameters(),lr=self.lr_c)

        # self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        # self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        #
        # self.actor_optim2 = torch.optim.Adam(self.actor2.parameters(), lr=self.lr_a)
        # self.critic_optim2 = torch.optim.Adam(self.critic2.parameters(), lr=self.lr_c)

        self.loss = nn.MSELoss()
        self.loss2 = nn.MSELoss()

    # def __init_model(self):
    #     self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
    #         obs_space=self._env.observation_space,
    #         act_space=self._env.action_space,
    #         model_cfgs=self._cfgs.model_cfgs,
    #         epochs=self._cfgs.train_cfgs.epochs,
    #     ).to(self._device)

    def get_action(self, s):
        if self._env_id=='kuhnpoker':#kuhn poker setting
            if (s[0,0]==0 and self._env.gametime%2 ==0) or (s[0,0]==1 and self._env.gametime%2 !=0):#P1 first
                print('P1',s[0,0])
                a = self.actor(s[0, :])  # state是两个拼接，故需要拆分
                a2 = torch.clamp(torch.randn(2, requires_grad=True), min=1e-8, max=1)
            else:
                print('P2',s[0,0])
                a2 = self.actor2(s[1, :])
                a=torch.clamp(torch.randn(2,requires_grad=True), min=1e-8, max=1)
        else:
            a = self.actor(s[0, :])  # state是两个拼接，故需要拆分
            a2 = self.actor2(s[1, :])

        dist = Categorical(a)
        action = dist.sample()  # 可采取的action
        log_prob = dist.log_prob(action)  # 每种action的概率

        dist2 = Categorical(a2)
        action2 = dist2.sample()  # 可采取的action
        log_prob2 = dist2.log_prob(action2)  # 每种action的概率

        return torch.tensor([[action], [action2]]), [log_prob, log_prob2]  # torch.tensor([[log_prob],[log_prob2]])

    def learn(
        self,
        log_prob_,
        s,
        s_,
        rew_
    )-> tuple[float, float, float]:
        log_prob,log_prob2=log_prob_
        s1, s2 = s
        s1_, s2_ = s_
        rew1, rew2 = rew_

        v = self.critic(s1)
        v_ = self.critic(s1_)
        critic_loss = self.loss(self.gamma * v_ + rew1, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        v = self.critic(s1)
        v_ = self.critic(s1_)
        td = self.gamma * v_ + rew1 - v  # 计算TD误差
        loss_actor = -log_prob * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()


        v2 = self.critic2(s2)
        v_2 = self.critic2(s2_)

        critic_loss2 = self.loss(self.gamma * v_2 + rew2, v2)
        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        self.critic_optim2.step()

        v2 = self.critic2(s2)
        v_2 = self.critic2(s2_)
        td2 = self.gamma * v_2 + rew2 - v2  # 计算TD误差
        loss_actor2 = -log_prob2 * td2.detach()
        # loss_actor2 = -td2#.detach()
        self.actor_optim2.zero_grad()
        loss_actor2.backward()
        self.actor_optim2.step()

    ##################
        ep_ret=0
        ep_cost=0
        ep_len=0
        # start_time = time.time()
        # self._logger.log('INFO: Start training')

        # for epoch in range(self._cfgs.train_cfgs.epochs):
        #     epoch_time = time.time()

            # rollout_time = time.time()
            # self._env.rollout(
            #     steps_per_epoch=self._steps_per_epoch,
            #     agent=self._actor_critic,
            #     buffer=self._buf,
            #     logger=self._logger,
            # )
            # self._logger.store({'Time/Rollout': time.time() - rollout_time})
            #
            # update_time = time.time()
            # self._update()
            # self._logger.store({'Time/Update': time.time() - update_time})

            # if self._cfgs.model_cfgs.exploration_noise_anneal:
            #     self._actor_critic.annealing(epoch)
            #
            # if self._cfgs.model_cfgs.actor.lr is not None:
            #     self._actor_critic.actor_scheduler.step()

            # self._logger.store(
            #     {
            #         'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
            #         'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
            #         'Time/Total': (time.time() - start_time),
            #         'Time/Epoch': (time.time() - epoch_time),
            #         'Train/Epoch': epoch,
            #         'Train/LR': (
            #             0.0
            #             if self._cfgs.model_cfgs.actor.lr is None
            #             else self._actor_critic.actor_scheduler.get_last_lr()[0]
            #         ),
            #     },
            # )

            # self._logger.dump_tabular()

            # save model to disk
            # if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
            #     epoch + 1
            # ) == self._cfgs.train_cfgs.epochs:
            #     self._logger.torch_save()

        # ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        # ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        # ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        # self._logger.close()
        # self._env.close()

        return ep_ret, ep_cost, ep_len

