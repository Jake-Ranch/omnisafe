import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import gym
import random
import matplotlib.pyplot as plt


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
        out = F.softmax(self.fc2(x), dim=-1)
        #print(s,out)
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

class Actor_Critic:
    def __init__(self, env):
        self.gamma = 1#0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4
        self.env = env
        self.action_dim = self.env.action_space  # 获取描述行动的数据维度
        self.state_dim = self.env.observation_space  # 获取描述环境的数据维度

        # self.action_dim = self.env.action_space.n  # 获取描述行动的数据维度
        # self.state_dim = self.env.observation_space.shape[0]  # 获取描述环境的数据维度

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

    def get_action(self, s):
        if (s[0,0]==0 and env.gametime%2 ==0) or (s[0,0]==1 and env.gametime%2 !=0):#P1 first
            print('P1',s[0,0])
            a = self.actor(s[0, :])  # state是两个拼接，故需要拆分
            a2 = torch.clamp(torch.randn(2, requires_grad=True), min=1e-8, max=1)
        else:
            print('P2',s[0,0])
            a2 = self.actor2(s[1, :])
            a=torch.clamp(torch.randn(2,requires_grad=True), min=1e-8, max=1)

        dist = Categorical(a)
        action = dist.sample()  # 可采取的action
        log_prob = dist.log_prob(action)  # 每种action的概率

        dist2 = Categorical(a2)
        action2 = dist2.sample()  # 可采取的action
        log_prob2 = dist2.log_prob(action2)  # 每种action的概率

        return torch.tensor([[action], [action2]]), [log_prob, log_prob2]  # torch.tensor([[log_prob],[log_prob2]])

    def learn(self, log_prob_, si, s_i, rew_i):
        log_prob,log_prob2=log_prob_
        s, s2 = si
        s_, s_2 = s_i
        rew, rew2 = rew_i

        v = self.critic(s)
        v_ = self.critic(s_)

        critic_loss = self.loss(self.gamma * v_ + rew, v)
        # critic_loss = rew-v+1
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        v = self.critic(s)
        v_ = self.critic(s_)

        td = self.gamma * v_ + rew - v  # 计算TD误差

        loss_actor = -log_prob * td.detach()
        # loss_actor = -td#.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()


        ##################################################
        # else:
        v2 = self.critic2(s2)
        v_2 = self.critic2(s_2)

        critic_loss2 = self.loss(self.gamma * v_2 + rew2, v2)
        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        self.critic_optim2.step()

        v2 = self.critic2(s)
        v_2 = self.critic2(s_)
        td2 = self.gamma * v_2 + rew2 - v2  # 计算TD误差
        loss_actor2 = -log_prob2 * td2.detach()
        # loss_actor2 = -td2#.detach()
        self.actor_optim2.zero_grad()
        loss_actor2.backward()
        self.actor_optim2.step()


class RockPaperScissors():
    def __init__(self):
        self.action_space = 3  # 独热编码
        self.gamename = 'RockPaperScissors'
        # self.gamma=0.99
        self.totalgametime = 200  # 总博弈次数
        self.actionhistory = 5  # 记录双方博弈历史
        # self.observation_space = self.actionhistory * 3 + 3
        self.observation_space = self.actionhistory * 7 + 3

    def reset(self):
        self.state1 = torch.zeros(1, self.observation_space)  # 十个历史数据，分别是双方的动作和奖惩
        self.state2 = torch.zeros(1, self.observation_space)  # 十个历史数据，分别是双方的动作和奖惩
        self.state1[0, -3:] = 1
        self.state2[0, -3:] = 1
        self.gametime = 0  # 博弈次数

        return torch.cat((self.state1, self.state2), dim=0).detach()

    # reward:
    #     R     P      S      A
    # R  0,0   1,-1  -1,1
    # P -1,1   0,0    1,-1
    # S  1,-1 -1,1    0,0

    # B

    def step(self, action_):  # 0=没出，1=R,2=P,3=S
        # action1, action2=action
        # action1=action1[0]
        # action2=action2[0]
        action1, action2 = action_

        self.gametime += 1
        done = 0
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


                # self.state1[0, :self.actionhistory * 3 - 1] = self.state1[0, 1:self.actionhistory * 3].clone()
                # self.state2[0, :self.actionhistory * 3 - 1] = self.state2[0, 1:self.actionhistory * 3].clone()
                # self.state1[0, self.actionhistory - 1] = action2
                # self.state2[0, self.actionhistory - 1] = action1
                # self.state1[0, self.actionhistory * 2 - 1] = action1
                # self.state2[0, self.actionhistory * 2 - 1] = action2


                # if action1 != 0 and action2 != 0:
                action1=action1+1
                action2=action2+1
                self.state1[0, -action2] += 1  # 记录对手的出招次数
                self.state2[0, -action1] += 1  # 记录对手的出招次数
                print(self.state1[0, -3:],self.state2[0, -3:])

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
                done = 1

            # self.state1[0, self.actionhistory * 3 - 1] = reward1
            # self.state2[0, self.actionhistory * 3 - 1] = reward2
            self.state1[0, self.actionhistory * 7 - 1] = reward1
            self.state2[0, self.actionhistory * 7 - 1] = reward2

            next_state1 = self.state1.clone()
            next_state2 = self.state2.clone()
            next_state1[0, -3:] = next_state1[0, -3:].clone() / torch.sum(self.state1[0, -3:]).clone()
            next_state2[0, -3:] = next_state2[0, -3:].clone() / torch.sum(self.state2[0, -3:]).clone()

            next_state = torch.cat((next_state1, next_state2), dim=0)
            reward = torch.tensor([[reward1], [reward2]])  # torch.cat((reward1, reward2), dim=0)
        return next_state.detach(), reward.detach(), done, False, {}

class KuhnPoker():
    def __init__(self):
        self.action_space = 2  # 独热编码
        self.name = 'KuhnPoker'
        self.totalgametime = 3  # 总博弈次数
        self.observation_space = 6

    def reset(self):
        first=random.randint(0,1)
        P1, P2 = torch.randperm(3)[:2]+1
        # self.state1 = torch.zeros(1, self.observation_space)  # 十个历史数据，分别是双方的动作和奖惩
        # self.state2 = torch.zeros(1, self.observation_space)  # 十个历史数据，分别是双方的动作和奖惩
        self.state1 = torch.FloatTensor([[first,0,0,0,0,P1]])  # 十个历史数据，分别是双方的动作和奖惩
        self.state2 = torch.FloatTensor([[1-first,0,0,0,0,P2]])  # 十个历史数据，分别是双方的动作和奖惩
        self.gametime = 0  # 博弈次数

        return torch.cat((self.state1, self.state2), dim=0).detach()

    def step(self, action_):  # 0=bet,1=pass
        action1,action2=action_
        self.gametime += 1
        done = 0
        reward1 = reward2 = reward = 0
        with torch.no_grad():
            if self.gametime <= self.totalgametime:
                if self.state1[0,0]==1:#P1后手bp
                    if self.gametime==1:
                        # print('P2未出，P2 bp')
                        self.state1[0, 1 + action2] = 1
                        self.state2[0, 3 + action2] = 1
                    elif self.gametime==2:
                        # print('P1未出，P1 bp')
                        self.state1[0, 3 + action1] = 1
                        self.state2[0, 1 + action1] = 1
                        P1p = self.state2[0, 2]
                        P1b = self.state2[0, 1]
                        P2p = self.state1[0, 2]
                        P2b = self.state1[0, 1]
                        if (P1p and P2p) or (P1p and P2b) or (P1b and P2b):
                            done=1
                    else:
                        done=1
                    if done:
                        # print('P1已出，等待P2决定')
                        P1 = self.state1[0, -1]
                        P2 = self.state2[0, -1]
                        P1p = self.state2[0, 2]
                        P1b = self.state2[0, 1]
                        P2p = self.state1[0, 2]
                        P2b = self.state1[0, 1]

                        if P1p and P2p:  # P1 P2 pass
                            reward = 1
                        elif (P2b or action2 == 0) and P1b:  # P1 P2 bet/ P2 pass P1 bet,P2 bet again
                            reward = 2
                        elif P2p and P1b and action2 == 1:  # P2 pass P1 bet,P2 pass again
                            reward = 1
                        elif P2b and P1p:  # P1 pass P2 bet,P2+1
                            reward = 1

                        if (P2p and P1b and action2==1):
                            reward2 = -reward
                            reward1 = reward
                        elif P1 < P2 or (P2b and P1p) :  # P1 pass P2 bet,P2+1
                            reward2 = reward
                            reward1 = -reward
                        else:  # P1 pass P2 bet
                            reward2 = -reward
                            reward1 = reward

                else:#P1先手
                    if self.gametime==1:
                        self.state2[0, 1 + action1] = 1
                        self.state1[0, 3 + action1] = 1
                    elif self.gametime==2:
                        self.state2[0, 3 + action2] = 1
                        self.state1[0, 1 + action2] = 1
                        P1p = self.state2[0, 2]
                        P1b = self.state2[0, 1]
                        P2p = self.state1[0, 2]
                        P2b = self.state1[0, 1]
                        if (P1p and P2p) or (P1b and P2p) or (P1b and P2b):
                            done=1
                    else:
                        done=1
                    if done:
                        P1 = self.state1[0, -1]
                        P2 = self.state2[0, -1]
                        P1p=self.state2[0, 2]
                        P1b=self.state2[0, 1]
                        P2p=self.state1[0, 2]
                        P2b=self.state1[0, 1]

                        if P1p and P2p:  # P1 P2 pass
                            reward = 1
                        elif (P1b or action1 == 0) and P2b:  # P1 P2 bet/ P2 pass P1 bet,P2 bet again
                            reward = 2
                        elif P1p and P2b and action1 == 1:  # P2 pass P1 bet,P2 pass again
                            reward = 1
                        elif P1b and P2p:  # P1 pass P2 bet,P2+1
                            reward = 1

                        if (P1p and P2b and action1==1):
                            reward1 = -reward
                            reward2 = reward
                        elif P1 > P2 or (P1b and P2p):  # P1 pass P2 bet,P2+1
                            reward1 = reward
                            reward2 = -reward
                        else:  # P1 pass P2 bet
                            reward1 = -reward
                            reward2 = reward
            else:
                done=1
            next_state = torch.cat((self.state1.clone(), self.state2.clone()), dim=0)

            reward_ = torch.tensor([[reward1], [reward2]])

        return next_state.detach(), reward_.detach(), done, False, {}


if __name__ == "__main__":
    env = RockPaperScissors()
    # env = KuhnPoker()
    model = Actor_Critic(env)  # 实例化Actor_Critic算法类
    reward = torch.zeros(2, 1)
    actionlist=torch.zeros(2,1)

    while True:
        inp=input('>>>')
        try:
            inp=int(inp)
        except:
            pass

        if type(inp)==int:
            for episode in range(inp):
                s = env.reset()  # 获取环境状态
                done = False  # 记录当前回合游戏是否结束
                ep_r = 0
                ep_r2 = 0
                while not done:
                    # 通过Actor_Critic算法对当前环境做出行动
                    a, log_prob = model.get_action(s)

                    s_, rew, done, _, _ = env.step(a)

                    # 计算当前reward
                    ep_r += rew

                    # 训练模型
                    model.learn(log_prob, s, s_, rew)

                    # 更新环境
                    s = s_
                reward = torch.cat((reward, ep_r), dim=1)
                print(f"episode:{episode} ep_r:{ep_r.tolist()} ")
            plt.plot(reward[0, :])
            plt.plot(reward[1, :])
            plt.show()

