import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu"

# 超参数
BATCH_SIZE = 32  # 批处理大小
LR = 0.01  # 学习率
EPSILON = 0.95  # 探索概率
GAMMA = 0.95  # 奖赏折扣
UPDATE_STEP = 50  # 目标网络更新步长
MEMORY_CAPACITY = 500  # 存储池容量
LR_MIN = 1e-5  # 最小学习率

N_STATES = 24  # 状态空间
N_ACTIONS = 20  # 动作空间
ENV_A_SHAPE = 0  # 动作空间的形状/维数

NODE = 100  # 隐藏层结点数


# 定义和实现DQN算法中神经网络模型，它用于估计每个状态下的动作值函数，以支持智能体在环境中做出合适的决策
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 三层线性层
        self.fc1 = nn.Linear(N_STATES, NODE)  # 从输入状态中提取特征
        self.fc1.weight.data.normal_(0, 0.1)  # 使用正态分布（均值为0，标准差为0.1）来初始化权重

        self.fc2 = nn.Linear(NODE, NODE)  # 隐藏层之间的连接
        self.fc2.weight.data.normal_(0, 0.1)

        self.fc3 = nn.Linear(NODE, N_ACTIONS)  # 输出层将根据隐藏层的表示输出每个动作的Q值估计
        self.fc3.weight.data.normal_(0, 0.1)

    # 神经网络模型的前向传播过程
    def forward(self, x):
        h1 = F.relu(self.fc1(x))  # 将输入x通过第一个全连接层并经过ReLU激活函数进行非线性变换，得到隐藏层的输出h1
        h2 = F.relu(self.fc2(h1))  # 将隐藏层的输出h1通过第二个全连接层并经过ReLU激活函数进行非线性变换，得到新的隐藏层输出h2
        output = self.fc3(h2)  # 将隐藏层的输出h2通过输出层，得到每个动作对应的Q值估计output
        return output  # 返回Q值作为网络的输出


class Agent:
    def __init__(self):
        self.online_net, self.target_net = Net().to(device), Net().to(device)  # 创建了两个神经网络模型
        self.learn_step_counter = 0  # 学习步数
        self.memory_counter = 0  # 记忆的数量

        # 就是一个二维数组，数组的每一行表示一个经验元组，即一个状态、一个下一个状态、一个动作、一个奖励和一个标志位（1表示完成，0表示未完成）
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2 + 1))  # numpy数组，用于存储状态、动作、奖励等信息
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LR)  # Adam 优化器，用于优化在线网络的参数
        self.loss_func = nn.SmoothL1Loss()  # 平滑的 L1 损失函数，用于衡量预测值和目标值之间的差异

    # 基于epsilon-greedy算法的动作选择策略
    def action(self, state, random=True):
        # 行方向扩充一个维度，以便于作为神经网络的输入
        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)
        random_num = np.random.uniform()  # 用于判断应该是探索（随机选择动作）还是利用（选择Q估计值最大的动作）

        # 探索  随机选择动作
        if random and (random_num < EPSILON):
            action = np.random.randint(0, N_ACTIONS)

        # 利用  选择Q估计值最大的动作
        else:
            actions_value = self.online_net.forward(state.to(device))  # 利用在线网络求得当前状态下所有动作的估计价值
            action = torch.max(actions_value, 1)[1].data.to('cpu').numpy()[0]  # 选择具有最大值的动作

        # 如果ENV_A_SHAPE不为0，则对输出动作进行形状变换，以适应环境的动作空间形状
        action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    # 智能体经验回放的存储函数，存储智能体与环境交互得到的经验数据
    def store(self, state, action, reward, next_state, done):
        # 接收智能体与环境交互得到的一个经验数据，将这些信息整合成一个经验元组transition
        if done:
            transition = np.hstack((state, action, reward, next_state, 0))  # 0 表示最终状态
        else:
            transition = np.hstack((state, action, reward, next_state, 1))  # 1 表示非最终状态

        # 将新的经验元组替换掉旧的经验数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition  # 赋给第index行
        self.memory_counter += 1

    # 用于更新神经网络的参数以优化Q值的估计
    def learn(self, algorithm="DDQN"):
        # 目标网络参数更新
        if self.learn_step_counter % UPDATE_STEP == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())  # 将在线网络的参数加载到目标网络上
        self.learn_step_counter += 1

        # 从经验回放缓冲区中随机采样一批数据作为训练数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 从 0 到 MEMORY_CAPACITY-1 的范围内，随机选择 BATCH_SIZE 个不重复的整数作为样本索引存储在 sample_index 中
        batch_memory = self.memory[sample_index, :]  # 根据sample_index中存储的索引，在memory中取出对应的行

        # 将之前获取的批量记忆数据 batch_memory 进行处理，并将其转换为适合深度学习模型训练的张量形式
        batch_state = torch.FloatTensor(batch_memory[:, :N_STATES]).to(device)
        batch_action = torch.LongTensor(batch_memory[:, N_STATES:N_STATES + 1].astype(int)).to(device)
        batch_reward = torch.FloatTensor(batch_memory[:, N_STATES + 1:N_STATES + 2]).to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:, -N_STATES - 1:-1]).to(device)
        batch_done = torch.FloatTensor(batch_memory[:, -1]).to(device)

        loss_value = 0  # 求误差，并反向传播更新网络权值

        # Q 值更新算法
        if algorithm == "DQN":
            # 在当前状态下，实际执行动作对应的 Q 值
            q_eval = self.online_net.forward(batch_state).gather(1, batch_action)

            # 预测下一个状态的 Q 值，detach 防止梯度传播到目标网络
            q_next = self.target_net.forward(batch_next_state).detach()

            '''
                计算目标 Q 值，近似贝尔曼方程，与Q-Learning更新公式不太一样
                ！！！！！目标Q值，即我们希望神经网络逼近的值！！！！！
            '''
            q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) * batch_done.view(BATCH_SIZE, 1)

            # 反向传播和更新网络参数(最后，将这个更新后的目标Q值 q_target 作为损失函数的目标，用于更新神经网络的参数)
            loss = self.loss_func(q_eval, q_target)  # 计算预测 Q 值与目标 Q 值之间的损失
            loss_value = float(loss)

            self.optimizer.zero_grad()  # 设置初始梯度为0
            loss.backward()  # 误差反向传播-- 计算出损失函数对所有参数的梯度
            self.optimizer.step()  # 更新权重--优化器会根据损失函数的梯度信息和设定的学习率等超参数来更新每个参数

        elif algorithm == "DDQN":
            '''
                1.在 DQN 中，使用目标网络预测下一个状态的 Q 值，并直接选择具有最大 Q 值的动作
                  然后再用这个 Q 值去估计目标 Q 值。
                2.而在 DDQN 中，首先使用在线网络预测下一个状态下各个动作的 Q 值，然后选择具有最大 Q 值的动作。
                  接着，再利用目标网络计算该动作对应的 Q 值。这样可以有效减轻对 Q 值的高估估计。
            '''
            actions_value = self.online_net.forward(batch_next_state)
            next_action = torch.unsqueeze(torch.max(actions_value, 1)[1], 1)

            q_eval = self.online_net.forward(batch_state).gather(1, batch_action)
            q_next = self.target_net.forward(batch_next_state).gather(1, next_action)
            q_target = batch_reward + GAMMA * q_next * batch_done.view(BATCH_SIZE, 1)

            # 反向传播和更新网络参数
            loss = self.loss_func(q_eval, q_target)
            loss_value = float(loss)

            self.optimizer.zero_grad()  # 设置初始梯度为0
            loss.backward()  # 误差反向传播
            self.optimizer.step()  # 更新权重

        return loss_value

    # 保存智能体的在线网络和目标网络的参数
    def save(self, file, episode=None, best=True):
        if episode is not None:
            torch.save(self.online_net.state_dict(), file + 'online_network_%d.pkl' % episode)
            torch.save(self.target_net.state_dict(), file + 'target_network_%d.pkl' % episode)
        if best:
            torch.save(self.online_net.state_dict(), file + 'online_network_best.pkl')
            torch.save(self.target_net.state_dict(), file + 'target_network_best.pkl')
