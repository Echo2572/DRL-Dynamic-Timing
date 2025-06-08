import gc
import os
import string
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget
from torch import cuda, load
from torch.backends import cudnn

import agent.dqn as ag
from environment.vissim import VisCom

import warnings

warnings.filterwarnings('ignore', message='Starting a Matplotlib GUI outside of the main thread will likely fail.')

device = 'cuda' if cuda.is_available() else 'cpu'  # 判断是否安装了cuda
CUDNN = cudnn.is_available()  # 判断是否安装了cuDNN

EPSILON_MAX = 0.99  # greedy 最大贪婪策略
EPSILON_MIN = 0.05  # greedy 最小贪婪策略
LR_MAX = 0.01  # 最大学习率
LR_MIN = 1e-5  # 最小学习率
EPISODE = 100  # 训练回合数
MAX_STEP = 42  # 最大回合步长
TEST_STEP = MAX_STEP  # 测试步长
TEST_FREQUENCY = 2 / 5  # 测试频率
ZEROREWARD = 38.509  # 延误 == 39.337s  <-->  奖励 == 0
CONVERGENCE_UP = 4  # 奖励收敛上限  (range:-10~10)
CONVERGENCE_LOW = -4  # 奖励收敛下限  (range:-10~10)
CONVERGENCE = int(MAX_STEP * 0.20) + 1  # 收敛计数器

# 超参数
LR = LR_MAX  # 学习率
EPSILON = EPSILON_MAX  # greedy 贪婪策略
GAMMA = 0.95  # 奖赏折扣
BATCH_SIZE = 32  # 批处理大小
UPDATE_STEP = 50  # 目标网络更新步长
MEMORY_CAPACITY = MAX_STEP * 10  # 存储池容量
EPSILON_DAMPING = (EPSILON_MIN / EPSILON) ** (1 / EPISODE)  # 探索衰减因子
LR_DAMPING = (LR_MIN / LR) ** (1 / EPISODE)  # 学习衰减因子

N_ACTIONS = 20
N_STATES = 24
ENV_A_SHAPE = 0
NODE = 100

ALGORITHM = "DQN"
LOSS = "SmoothL1Loss"
OPTIM = "Adam"
ACTIVATE = "relu"

train_file = './model/txt/train_record.txt'
train_test_file = './model/txt/test_record.txt'
drl_test_file = './model/txt/test_drl_record.txt'
status_file = './model/txt/test_status_record.txt'
online_network = './model/pkl/online_network_best.pkl'
test_network = './online_network_best.pkl'
# test_network = './model/pkl/online_network_best.pkl'
test_fix_file = "./fix/txt/test_fix_record.txt"

# vssim 仿真路网文件路径
net_path = "E:\\Vissim-Python-Qt\\resource\\vissim\\net\\net.inp"
# vissim仿真设置: 最大仿真时长，仿真速度、模拟分辨率、控制频率、最大回合步长、图图形模式、3D模式
simulation = [999999, 0, 1, 1, 42, True, False]
model = 3


class Sim(QWidget):
    # 定义信号，在不同部件或线程之间进行通信
    SimInfoEmit = pyqtSignal(str)
    RateProgressInfoEmit = pyqtSignal(int)
    RemainTimeInfoEmit = pyqtSignal(int, int, int)

    def __init__(self):
        super().__init__()
        self.stop_flag = False

    # 获取系统中可用的磁盘列表
    @staticmethod
    def get_disklist():
        disk_list = []
        for c in string.ascii_uppercase:
            disk = c + ':'
            if os.path.isdir(disk):
                disk_list.append(disk)
        return disk_list

    # 在指定路径下搜索文件
    @staticmethod
    def search_file(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                file_path = os.path.abspath(os.path.join(root, f))
                if file_path.endswith("vissim.exe"):
                    return file_path
        return None

    # 运行vissim软件
    def find_vissim(self):
        for disk in self.get_disklist():
            path = self.search_file(disk + r"\Vissim4.3\Exe")

            if path is not None:
                print("Vissim is Found!")
                return path
            else:
                pass
        else:
            print("Not Found Vissim!")

    # 获取vissim环境
    def get_vissim_env(self, net_path, simulation, plans):
        path = self.find_vissim()  # 寻找vissim
        if path is not None:
            env = VisCom(path, net_path, simulation, plans)  # 初始化vissim环境
            return env
        else:
            print("Failed to init the environment!")

    # 智能体参数重赋值
    @staticmethod
    def agent_revalue():
        ag.device = device
        ag.BATCH_SIZE = BATCH_SIZE
        ag.LR = LR
        ag.EPSILON = EPSILON
        ag.GAMMA = GAMMA
        ag.UPDATE_STEP = UPDATE_STEP
        ag.MEMORY_CAPACITY = MEMORY_CAPACITY
        ag.LR_MIN = LR_MIN

        ag.N_STATES = N_STATES
        ag.N_ACTIONS = N_ACTIONS
        ag.ENV_A_SHAPE = ENV_A_SHAPE

        ag.NODE = NODE

    # 奖赏函数
    @staticmethod
    def get_reward(delay, ZEROREWARD):
        return round(ZEROREWARD - delay, 3)  # 四舍五入并保留三位小数

    # 保存训练日志
    @staticmethod
    def save_log(net_path, simulation, plans):
        now = time.strftime("time:%Y-%m-%d-%H:%M:%S\n", time.localtime(time.time()))
        content = ""
        content += now
        content += "net_path:{}\n".format(net_path)
        content += "simulation:{}\n".format(simulation)
        content += "plans:\n"

        for i in range(len(plans)):
            content += str(plans[i])
            content += "\n"

        content += "device:{}\n".format(device)
        content += "LR:{}\n".format(LR)
        content += "LR_MIN:{}\n".format(LR_MIN)
        content += "LR_MAX:{}\n".format(LR_MAX)
        content += "EPSILON:{}\n".format(EPSILON)
        content += "EPSILON_MIN:{}\n".format(EPSILON_MIN)
        content += "EPSILON_MAX:{}\n".format(EPSILON_MAX)
        content += "GAMMA:{}\n".format(GAMMA)
        content += "EPISODE:{}\n".format(EPISODE)
        content += "MEMORY_CAPACITY:{}\n".format(MEMORY_CAPACITY)
        content += "BATCH_SIZE:{}\n".format(BATCH_SIZE)
        content += "UPDATE_STEP:{}\n".format(UPDATE_STEP)
        content += "TEST_FREQUENCY:{}\n".format(TEST_FREQUENCY)
        content += "CONVERGENCE_UP:{}\n".format(CONVERGENCE_UP)
        content += "CONVERGENCE_LOW:{}\n".format(CONVERGENCE_LOW)
        content += "ZEROREWARD:{}\n".format(ZEROREWARD)
        content += "CONVERGENCE:{}\n".format(CONVERGENCE)
        content += "ALGORITHM:{}\n".format(ALGORITHM)
        content += "LOSS:{}\n".format(LOSS)
        content += "OPTIM:{}\n".format(OPTIM)
        content += "ACTIVATE:{}\n".format(ACTIVATE)
        content += "N_ACTIONS:{}\n".format(N_ACTIONS)
        content += "N_STATES:{}\n".format(N_STATES)
        content += "ENV_A_SHAPE:{}\n".format(ENV_A_SHAPE)
        content += "NODE:{}\n".format(NODE)

        with open('./model/txt/tarin_log.txt', 'w') as f:
            f.write(content)

    # 创建仿真环境--设置观测空间和动作空间
    def create_environment(self, net_path, simulation, plans):
        from gym.spaces import Box, Discrete

        env = self.get_vissim_env(net_path, simulation, plans)
        env.action_space = Discrete(len(plans))  # 动作空间描述

        global N_ACTIONS, N_STATES
        N_STATES = len(env.reset())  # 状态空间描述
        N_ACTIONS = env.action_space.n

        low = np.array([0 for _ in range(N_STATES)], dtype=np.float32)
        high = np.array([1000 for _ in range(N_STATES)], dtype=np.float32)

        env.observation_space = Box(low, high, dtype=np.float32)
        ag.N_STATES = env.observation_space.shape[0]
        ag.N_ACTIONS = N_ACTIONS

        if isinstance(env.action_space.sample(), int):
            ag.ENV_A_SHAPE = 0
        else:
            ag.ENV_A_SHAPE = env.action_space.sample().shape

        print("Create environment successfully!")
        return env

    # 可视化绘制训练数据记录文件
    @staticmethod
    def draw_train_record(file1, file2):
        font = {'family': 'SimSun', 'weight': 'bold', 'size': '16'}
        plt.rc('font', **font)
        plt.rc('axes', unicode_minus=False)

        # 绘制训练曲线
        names = ["episode", "step", "epsilon", "learn_rate", "convergence", "delay", "reward", "loss"]
        data = pd.read_csv(file1, sep="\s+", names=names)

        delay = list(data["delay"].values)
        reward = list(data["reward"].values)
        loss = list(data["loss"].values)
        convergence = list(data["convergence"].values)
        epsilon = list(data["epsilon"].values)
        learn_rate = list(data["learn_rate"].values)

        """①延误"""
        # 绘图
        x = np.linspace(0, len(delay), len(delay))
        plt.plot(x, delay, "k")

        # 设置坐标轴名称
        plt.xlabel("Episode", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Delay", fontproperties="Times New Roman", size=10.5)

        # 设置图例
        legend = ["delay"]
        plt.legend(legend, loc="best", frameon=False)

        plt.savefig("./model/png/train_delay.png", dpi=600)  # 保存图片
        plt.close()  # 关闭绘图

        """②奖赏"""
        # 绘图
        x = np.linspace(0, len(reward), len(reward))
        plt.plot(x, reward, "k")

        # 设置坐标轴名称
        plt.xlabel("Episode", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Reward", fontproperties="Times New Roman", size=10.5)

        # 设置图例
        legend = ["reward"]
        plt.legend(legend, loc="best", frameon=False)

        plt.savefig("./model/png/train_reward.png", dpi=600)  # 保存图片
        plt.close()  # 关闭绘图

        """③损失"""
        # 绘图
        idx = 0
        for i in range(len(loss)):
            if loss[i] != 0:
                idx = i
                break

        loss = loss[idx::]
        x = np.linspace(0, len(loss), len(loss))
        plt.plot(x, loss, "k")

        # 设置坐标轴名称
        plt.xlabel("Episode", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Loss", fontproperties="Times New Roman", size=10.5)

        # 设置图例
        legend = ["loss"]
        plt.legend(legend, loc="best", frameon=False)

        plt.savefig("./model/png/train_loss.png", dpi=600)  # 保存图片
        plt.close()  # 关闭绘图

        """④探索率"""
        # 绘图
        x = np.linspace(0, len(epsilon), len(epsilon))
        plt.plot(x, epsilon, "k")

        # 设置坐标轴名称
        plt.xlabel("Episode", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Epsilon", fontproperties="Times New Roman", size=10.5)

        # 设置图例
        legend = ["epsilon"]
        plt.legend(legend, loc="best", frameon=False)

        plt.savefig("./model/png/train_epsilon.png", dpi=600)  # 保存图片
        plt.close()  # 关闭绘图

        """⑤学习率"""
        # 绘图
        x = np.linspace(0, len(learn_rate), len(learn_rate))
        plt.plot(x, learn_rate, "k")

        # 设置坐标轴名称
        plt.xlabel("Episode", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Learn_Rate", fontproperties="Times New Roman", size=10.5)

        # 设置图例
        legend = ["learn_rate"]
        plt.legend(legend, loc="best", frameon=False)

        plt.savefig("./model/png/train_learn_rate.png", dpi=600)  # 保存图片
        plt.close()  # 关闭绘图

        # 绘制训练测试曲线
        names = ["episode", "step", "delay", "reward"]
        data = pd.read_csv(file2, sep="\s+", names=names)
        delay = list(data["delay"].values)
        reward = list(data["reward"].values)

        """⑥收敛计数器"""
        # 绘图
        x = np.linspace(0, len(convergence), len(convergence))
        plt.plot(x, convergence, "k")

        # 设置坐标轴名称
        plt.xlabel("Episode", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Convergence", fontproperties="Times New Roman", size=10.5)

        # 设置图例
        legend = ["convergence"]
        plt.legend(legend, loc="best", frameon=False)

        plt.savefig("./model/png/train_convergence.png", dpi=600)  # 保存图片
        plt.close()  # 关闭绘图

        """⑦测试延误"""
        # 绘图
        x = np.linspace(0, len(delay), len(delay))
        plt.plot(x, delay, "k")

        # 设置坐标轴名称
        plt.xlabel("Episode", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Delay", fontproperties="Times New Roman", size=10.5)

        # 设置图例
        legend = ["delay"]
        plt.legend(legend, loc="best", frameon=False)

        plt.savefig("./model/png/train_test_delay.png", dpi=600)  # 保存图片
        plt.close()  # 关闭绘图

        """⑧测试奖赏"""
        # 绘图
        x = np.linspace(0, len(reward), len(reward))
        plt.plot(x, reward, "k")

        # 设置坐标轴名称
        plt.xlabel("Episode", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Reward", fontproperties="Times New Roman", size=10.5)

        # 设置图例
        legend = ["reward"]
        plt.legend(legend, loc="best", frameon=False)

        plt.savefig("./model/png/train_test_reward.png", dpi=600)  # 保存图片 
        plt.close()  # 关闭绘图

    # 可视化绘制DRL测试数据记录文件(只有一张图)
    @staticmethod
    def draw_test_record(file3, file4):
        font = {'family': 'SimSun', 'weight': 'bold', 'size': '16'}
        plt.rc('font', **font)
        plt.rc('axes', unicode_minus=False)

        # 绘制DRL训练结果测试曲线
        names = ["step", "plan", "delay", "reward"]
        data = pd.read_csv(file3, sep="\s+", names=names)
        delay = list(data["delay"].values)

        mean_delay = round(sum(delay) / len(delay), 3)
        names = ["plan", "step", "delay"]
        drl_plan = list(data["plan"].values)
        fix_data = pd.read_csv(file4, sep="\s+", names=names)
        fix_plan = list(fix_data["plan"].values)
        fix_delay = list(fix_data["delay"].values)
        compar = list(map(lambda x: round((x - mean_delay) / x * 100, 2), fix_delay))

        """测试延误"""
        # 绘图
        x = np.linspace(0, len(delay), len(delay))
        mean_delay = np.mean(delay)
        plt.plot(x, delay, color='black', marker='D', linestyle='-', linewidth='1.0')
        plt.plot(x, [mean_delay for _ in range(len(x))], color='gray', linestyle='--')

        # 设置坐标轴名称
        plt.xlabel("Step", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Delay", fontproperties="Times New Roman", size=10.5)

        # 设置图例
        legend = ["drl delay", "mean delay line"]
        plt.legend(legend, loc="best", frameon=False)

        plt.savefig("./model/png/drl_test_delay.png", dpi=600)  # 保存图片
        plt.close()  # 关闭绘图

        """方案选择频率"""
        se = pd.Series(drl_plan)
        plan_num = len(fix_plan)
        proportitionDict = dict(se.value_counts(normalize=True))
        plan_freq = []

        for i in range(plan_num):
            try:
                plan_freq.append(proportitionDict[i])
            except:
                plan_freq.append(0)
        x = np.linspace(0, len(plan_freq) - 1, len(plan_freq))
        plt.bar(x, plan_freq, 0.5, color="gray", edgecolor="k")

        plt.xlabel("Plan Index", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Frequency", fontproperties="Times New Roman", size=10.5)

        plt.savefig("./model/png/plan_frequency.png", dpi=600)  # 保存图片
        plt.close()

        """对比固定配时延误"""
        x = np.linspace(0, len(compar) - 1, len(compar))
        plt.plot(x, fix_delay, color='black', marker='D', linestyle='-')
        plt.plot(x, [mean_delay for _ in range(len(fix_plan))], color='gray', marker='*', linestyle='--')

        plt.xlabel("Plan Index", fontproperties="Times New Roman", size=10.5)
        plt.ylabel("Delay", fontproperties="Times New Roman", size=10.5)

        plt.savefig("./model/png/compare_fixed_plan.png", dpi=600)  # 保存图片
        plt.close()

    # 测试智能体
    def test(self, env, agent, online_net):
        print(1)
        agent.online_net.load_state_dict(load(online_net, map_location=device))  # 加载目标网络

        print(2)
        # 初始化参数
        test_start = time.perf_counter()
        test_delay_record = []
        test_reward_record = []

        # 重置环境获取初始交通流状态
        state = env.reset()

        # 热身时间
        for i in range(5):
            state, reward, done, info = env.step(0)

        # 仿真运行指定个周期
        for step in range(42):
            print("Test--" + str(step))
            action = agent.action(state, random=False)  # 智能体由交通流状态获取配时动作方案
            next_state, reward, done, info = env.step(action)  # vissim环境采取动作运行一周期，得到下一周期的状态信息

            # 重定义奖励
            delay = reward
            redefine_reward = self.get_reward(delay, ZEROREWARD)

            # 更新状态、奖励、平均延误、当前回合仿真步数
            state = next_state
            test_delay_record.append(delay)
            test_reward_record.append(redefine_reward)

            # 保存状态信息
            with open(status_file, 'a+') as f:
                record = "%s\t\n" % state
                f.write(record)

            # 保存测试记录信息
            with open(drl_test_file, 'a+') as f:
                record = "%-5s\t%-5s\t%-5s\t%-5s\t\n" % (
                    str(step + 1), str(action + 1), str(delay), str(redefine_reward))
                f.write(record)

        # 输出最佳网络的测试奖励值和延误值
        test_mean_delay = sum(test_delay_record) / len(test_delay_record)
        test_mean_reward = sum(test_reward_record) / len(test_reward_record)
        info = 'test step: {}, test_mean_delay: {}, test_mean_reward: {}'.format(TEST_STEP, round(test_mean_delay, 3),
                                                                                 round(test_mean_reward, 3))
        self.simulation_information(info)

        # 输出测试时间
        test_time = time.perf_counter() - test_start
        h, ss = divmod(test_time, 3600)
        m, s = divmod(ss, 60)
        info = "complete test time: {} second, that is {} hour, {} minute, {} second".format(test_time, h, m, s)
        self.simulation_information(info)

    # 训练智能体
    def train(self, env, agent, plans):
        # 保存初始网络
        agent.save('./model/pkl/')
        info = "Save the original neural network ......"
        self.simulation_information(info)
        best_reward = -10

        # 开始训练(控制整个训练过程的重复次数)
        for episode in range(100):
            if episode > 0 and episode % 19 == 0:
                env.close()
                del env  # 释放旧的 VISSIM 资源
                gc.collect()  # 释放内存
                time.sleep(3)
                env = self.create_environment(net_path, simulation, plans)
            # 启动当前回合训练
            print("EPISODE--" + str(episode))
            episode += 1
            start = time.perf_counter()  # 返回一个高性能时间戳
            delay_record = []  # 每个仿真周期中的延误
            reward_record = []  # 每个仿真周期中的奖励
            loss_record = []  # 每个仿真周期中的损失
            loss = 0  # 损失值
            success = 0  # 成功计数器
            max_success = 0  # 最大成功计数
            fail = 0  # 失败计数器
            step_count = 0  # 步数计数器
            convergence_test = False  # 收敛标志

            # 输出当前仿真进度
            info = "Start of the {} Episode Train".format(episode)
            self.simulation_information(info)
            rate_progress = int(episode / EPISODE * 100)  # 训练进度(第episode个训练回合，总共EPISODE个回合)
            self.RateProgressInfoEmit.emit(rate_progress)
            state = env.reset()  # 状态空间(一维数组)--重建状态空间

            # 热身时间 -- 1.稳定环境状态  2.填充经验池  3.初始化网络权重
            for i in range(5):
                state, reward, done, info = env.step(0)

            # 运行当前回合(控制每个训练回合中仿真周期的数量)
            for step in range(42):
                print("     STEP--" + str(step))
                action = agent.action(state)  # 由交通流状态获取配时动作方案(ε-greedy策略选择动作)
                next_state, reward, done, info = env.step(action)  # vissim环境采取动作运行一周期，得到下一周期的状态信息

                # 重定义奖励(重新定义奖励：比如delay达到某个阈值时，被赋予零奖励)
                delay = reward
                redefine_reward = self.get_reward(delay, ZEROREWARD)

                # 判断收敛条件
                if redefine_reward >= CONVERGENCE_UP:
                    success += 1
                    if success > max_success:
                        max_success = success
                else:
                    success = 0

                if success >= CONVERGENCE:
                    convergence_test = True
                    done = True

                if redefine_reward <= CONVERGENCE_LOW:
                    fail += 1
                else:
                    fail = 0

                if fail >= CONVERGENCE:
                    done = True

                # 存储样本到经验池
                agent.store(state, action, redefine_reward, next_state, done)

                # 智能体进行学习(经验池中收集足够的经验之后才开始进行学习)
                if agent.memory_counter > MEMORY_CAPACITY:
                    loss = agent.learn(ALGORITHM)

                # 更新状态、奖励、平均延误、当前回合训练步数
                state = next_state
                loss_record.append(loss)
                delay_record.append(delay)
                reward_record.append(redefine_reward)
                step_count += 1

                # 判断当前回合仿真结束标志
                if done:
                    break

            # 输出并保存当前回合数、回合总步数、探索概率、学习率、平均奖励、平均延误、平均损失、最大收敛次数
            mean_delay = sum(delay_record) / len(delay_record)
            mean_reward = sum(reward_record) / len(reward_record)
            mean_loss = sum(loss_record) / len(loss_record)
            info = "episode: {}, step: {}, epsilon: {}, lr: {}, convergence: {}, delay: {}, reward: {}, loss: {}".format(
                episode, step_count, ag.EPSILON, ag.LR, max_success, round(mean_delay, 3), round(mean_reward, 3),
                round(mean_loss, 3))
            self.simulation_information(info)

            # 保存训练回合记录文件
            with open(train_file, 'a+') as f:
                record = "%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t\n" % (
                    str(episode), str(step_count), str(round(ag.EPSILON, 3)), str(round(ag.LR, 3)), str(max_success),
                    str(round(mean_delay, 3)), str(round(mean_reward, 3)), str(round(mean_loss, 3)))
                f.write(record)

            # 逐渐衰减探索率
            if ag.EPSILON > EPSILON_MIN:
                # 衰减方法--余弦衰减
                x = episode / EPISODE * np.pi
                y = EPSILON_MIN + (np.cos(x) + 1) / 2 * (EPSILON_MAX - EPSILON_MIN)
                ag.EPSILON = y

            # 输出预计剩余训练时间
            train_episode_time = time.perf_counter() - start
            remain_time = train_episode_time * int((EPISODE - episode) * (1 + TEST_FREQUENCY))
            h, ss = divmod(remain_time, 3600)  # 返回商和余数，商为小时，余数为剩余的秒数
            m, s = divmod(ss, 60)  # 计算分钟和秒数
            h, m, s = int(h), int(m), int(s)
            info = "episode {} train time: {} second, remain simulation time: " "{:0>2d} hour, {:0>2d} minute, {:0>2d} second".format(
                episode, train_episode_time, h, m, s)
            self.simulation_information(info)
            self.RemainTimeInfoEmit.emit(h, m, s)

            # 测试网络性能
            c1 = convergence_test
            # c2 = (episode % max(1, int(EPISODE * TEST_FREQUENCY)) == 0)  # 是否是测试的时机--TEST_FREQUENCY可以表示每训练多少个回合或步骤后进行一次测试
            c2 = (episode % 2 == 0)
            c3 = (episode == EPISODE)
            c4 = (episode == 1)

            # 是进行测试的时机
            if c1 or c2 or c3 or c4:
                # 初始化参数
                test_delay_record = []
                test_reward_record = []

                # 重启环境并获取初始交通流状态
                state = env.reset()

                # 热身时间
                for i in range(2):
                    state, reward, done, info = env.step(0)

                # 运行指定个仿真周期 (用于测试)
                for step in range(5):
                    print("     Train-Test-step" + str(step))
                    action = agent.action(state, random=False)  # 由交通流状态获取配时动作方案
                    next_state, reward, done, info = env.step(action)  # vissim环境采取动作运行一周期，得到下一周期的状态信息

                    # 重定义奖励
                    delay = reward
                    redefine_reward = self.get_reward(delay, ZEROREWARD)
                    test_delay_record.append(delay)
                    test_reward_record.append(redefine_reward)

                    # 更新状态
                    state = next_state

                # 输出当前回合、测试平均延误、测试平均奖励
                test_mean_delay = sum(test_delay_record) / len(test_delay_record)
                test_mean_reward = sum(test_reward_record) / len(test_reward_record)
                info = 'episode: {}, test_mean_delay: {}, test_mean_reward: {}'.format(episode,
                                                                                       round(test_mean_delay, 3),
                                                                                       round(test_mean_reward, 3))
                self.simulation_information(info)

                # 保存测试回合记录文件
                with open(train_test_file, 'a+') as f:
                    record = "%-5s\t%-5s\t%-5s\t%-5s\t\n" % (
                        str(episode), str(TEST_STEP), str(round(test_mean_delay, 3)), str(round(test_mean_reward, 3)))
                    f.write(record)

                # 保存历史训练最优网络模型
                if test_mean_reward > best_reward:
                    best_reward = test_mean_reward
                    agent.save('./model/pkl/', episode)

                # 绘制延误、奖赏曲线
                self.draw_train_record(train_file, train_test_file)

    # 实时仿真信息
    def simulation_information(self, info):
        self.SimInfoEmit.emit(info)

    # 运行仿真
    def run(self, net_path, simulation, plans):
        global state

        program_start = time.perf_counter()  # 记录程序启动时间
        info = "Create vissim simulation environment......"
        self.simulation_information(info)
        env = self.create_environment(net_path, simulation, plans)  # 创建vissim仿真环境

        # 测试固定配时
        if model == 1:
            pass
        # 训练深度强化配时
        elif model == 2:
            # 保存训练日志
            info = "Save the training log file......"
            self.simulation_information(info)
            self.save_log(net_path, simulation, plans)

            # 重赋值智能体参数
            info = "Agent boot......"
            self.simulation_information(info)
            self.agent_revalue()
            my_agent = ag.Agent()  # 定义智能体

            # 训练智能体
            info = "Training agent......"
            self.simulation_information(info)
            print("Training agent......")
            self.train(env, my_agent, plans)

            # 测试最佳训练网络的性能
            info = "Testing agent......"
            self.simulation_information(info)
            print("Testing agent......")
            self.test(env, my_agent, online_network)

            # 绘制延误、奖赏曲线
            info = "Draw the training data image......"
            self.simulation_information(info)
            print("Draw the training data image......")
            self.draw_train_record(train_file, train_test_file)
            self.draw_test_record(drl_test_file, test_fix_file)

            print("Finish!")

        # 测试深度强化配时
        elif model == 3:
            # 重赋值智能体参数
            info = "Agent boot......"
            self.simulation_information(info)
            self.agent_revalue()
            my_agent = ag.Agent()  # 定义智能体

            # 测试最佳训练网络的性能
            info = "Testing agent......"
            self.simulation_information(info)
            self.test(env, my_agent, test_network)

            # 绘制延误、奖赏曲线
            info = "Draw the testing data image......"
            self.simulation_information(info)
            self.draw_test_record(drl_test_file, test_fix_file)

        program_end = time.perf_counter()  # 记录程序结束时间

        # 输出程序运行时间
        info = "program run time: %d second" % (program_end - program_start)
        self.simulation_information(info)
