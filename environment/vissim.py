import os
import time
from random import randint

import numpy as np
import psutil
from gym.spaces import Box, Discrete
from win32com.client import Dispatch


# noinspection SpellCheckingInspection
class VisCom:
    def __init__(self, program_file, net_file, simulation, plans):
        self.file = net_file.replace("/", "\\")  # 路网文件路径

        # vissim应用程序启动文件
        self.program = program_file
        self.simulation_para = simulation
        self.plans = plans

        self.action_space = Discrete(len(self.plans))  # 动作空间描述

        # 状态空间描述
        low = np.array([0 for _ in range(10)], dtype=np.float32)
        high = np.array([1000 for _ in range(10)], dtype=np.float32)
        self.observation_space = Box(low, high, dtype=np.float32)  # Box 表示固定上下界的 n 维连续空间

    """Gym高级API封装"""

    # 返回一个状态空间
    def reset(self):
        self.start()
        flow = [0 for _ in range(self.data_collections.Count)]  # 车流量
        speed = [0 for _ in range(self.data_collections.Count)]  # 车速
        queue = [0 for _ in range(self.queue_counters.Count)]  # 排队长度
        observation = flow + speed + queue
        return observation  # 本质是一维数组

    # 仿真运行一周期
    def step(self, action):
        plan = self.plans[action]
        cycle_time = plan[0]

        if cycle_time == 0:
            return 0, 0, 0, 0
        else:
            self.sc[0].SetAttValue("CYCLETIME", cycle_time)
            self.control_signal_group(plan)  # 根据 plan 设置信号灯的红绿灯结束时间

            if cycle_time is None:
                cycle_time = self.sc[0].AttValue("CYCLETIME")

            last_elapsed_time = self.simulation.AttValue("ELAPSEDTIME")
            stop_time = cycle_time + last_elapsed_time

            while True:
                elapsed_time = self.simulation.AttValue("ELAPSEDTIME")
                if stop_time > elapsed_time >= 0:
                    self.simulation.RunSingleStep()
                else:
                    self.ct = self.sc[0].AttValue("CYCLETIME")
                    self.offset = self.sc[0].AttValue("OFFSET")
                    self.elapsed_time = self.simulation.AttValue("ELAPSEDTIME")
                    break

            flow = list(map(lambda x: int(x), self.get_flow_collections_detector()))
            speed = list(map(lambda x: round(x, 2), self.get_speed_collections_detector()))
            queue = list(map(lambda x: round(x, 1), self.get_queue_counters_detector()))
            delay = list(map(lambda x: round(x, 2), self.get_delay_times_detector()))

            sys_time = time.strftime("%H:%M:%S")
            sim_time = int(self.elapsed_time)
            info = "%-8s  %-6s  %-32s %-56s %-52s %-7s %-4s" % (
            str(sys_time), str(sim_time), str(flow), str(speed), str(queue), str(delay), str(action))
            observation = flow + speed + queue
            reward = delay[0]

            if sim_time >= self.sim_stop_time - cycle_time:
                done = True
            else:
                done = False

            return observation, reward, done, info

    # 可视化vissim实时仿真画面
    def render(self):
        self.graphics.SetAttValue("VISUALIZATION", True)

    # 关闭环境，退出vissim仿真，并清除内存
    def close(self):
        self.end()

    # 设置随机数种子
    def seed(self, seed=42):
        self.random_seed = seed
        self.simulation.RandomSeed = self.random_seed

    """Vissim运行控制API"""

    # 启动仿真--初始化一些参数
    def start(self):
        # 仿真接口
        self.Vissim = Dispatch("VISSIM.Vissim.430")  # 创建COM对象的一种机制，它允许我们在Python中与其他应用程序进行交互，并调用其提供的功能
        self.Vissim.LoadNet(self.file)
        self.simulation = self.Vissim.Simulation
        self.graphics = self.Vissim.Graphics
        self.Net = self.Vissim.Net
        self.links = self.Net.Links
        self.inputs = self.Net.VehicleInputs
        self.vehicles = self.Net.Vehicles
        self.controllers = self.Net.SignalControllers
        self.groups = self.controllers(1).SignalGroups
        self.data_collections = self.Net.DataCollections  # DataCollections 是 VISSIM 网络中的一个数据集合，是 VISSIM 软件自身提供的接口之一
        self.travel_times = self.Net.TravelTimes
        self.delays = self.Net.Delays
        self.queue_counters = self.Net.QueueCounters

        # 仿真参数
        self.simulation.SetAttValue("PERIOD", self.simulation_para[0])
        self.sim_stop_time = self.simulation_para[0]
        self.simulation.Speed = self.simulation_para[1]
        self.simulation.Resolution = self.simulation_para[2]
        self.simulation.ControllerFrequency = self.simulation_para[3]
        self.simulation.RandomSeed = self.simulation_para[4]
        self.graphics.SetAttValue("VISUALIZATION", self.simulation_para[5])
        self.graphics.SetAttValue("3D", self.simulation_para[6])

        # 评价
        self.eval = self.Vissim.Evaluation
        self.qceval = self.eval.QueueCounterEvaluation
        self.dceval = self.eval.DataCollectionEvaluation
        self.deval = self.eval.DelayEvaluation
        self.tteval = self.eval.TravelTimeEvaluation
        self.linkeval = self.eval.LinkEvaluation
        self.eval.SetAttValue("DataCollection", True)
        self.eval.SetAttValue("TRAVELTIME", True)
        self.eval.SetAttValue("DELAY", True)
        self.eval.SetAttValue("QUEUECOUNTER", True)
        # self.eval.SetAttValue("LINK", True)
        self.qceval.SetAttValue("FILE", True)
        self.dceval.SetAttValue("FILE", True)
        self.deval.SetAttValue("FILE", True)
        self.tteval.SetAttValue("FILE", True)
        # self.linkeval.SetAttValue("FILE", True)

        # 信号控制机、信号灯组
        self.sc, self.sg = [], []

        # 检测器、保存数据
        self.tt, self.travel_time = [], []
        self.dt, self.delay = [], []
        self.dc, self.vel, self.speed = [], [], []
        self.qc, self.queue_length = [], []

        # 车流
        self.ip, self.ip_flow = [], []

        # 信号周期、相位差（周期延迟时间）、运行时间
        self.ct = 0
        self.offset = 0
        self.elapsed_time = 0

        # 随机数种子
        self.random_seed = self.simulation.RandomSeed
        if self.Vissim is None:
            self.run_vissim_exe()
            while True:
                try:
                    self.Vissim = Dispatch("VISSIM.Vissim")
                    break
                except:
                    pass

        if self.random_seed == 0:
            self.simulation.RandomSeed = randint(1, 9999)

        # 设置检测器
        self.set_signal_controller()
        self.set_signal_group()
        self.set_data_collections_detector()
        self.set_travel_times_detector()
        self.set_delay_times_detector()
        self.set_queue_counters_detector()

    # 停止仿真
    def stop(self):
        if self.Vissim is not None:
            self.simulation.Stop()

    # 结束仿真
    def end(self):
        if self.Vissim is not None:
            # 正在运行中，停止仿真
            if self.simulation.AttValue("ELAPSEDTIME") > 0:
                self.stop()
                self.Vissim.Exit()
                self.Vissim = None
            # 没有运行仿真，直接退出程序
            else:
                self.Vissim.Exit()
                self.Vissim = None

    # 检测vissim是否运行
    def detect_vissim(self):
        try:
            pids = psutil.pids()
            for pid in pids:
                p = psutil.Process(pid)
                process_name = p.name()
                if "vissim" in process_name:
                    return True
            else:
                return False
        except:
            return False

    # 运行vissim应用程序
    def run_vissim_exe(self):
        # 检测vissim是否运行
        flag = self.detect_vissim()
        if flag:
            # 强制结束当前vissim进程
            os.system("taskkill /F /IM vissim.exe")
        # 自动重启vissim软件
        os.system(r'RunAsDate.exe /movetime 03\05\2008 21:04:00 "%s"' % self.program)

        while True:
            # 等待重启成功
            flag = self.detect_vissim()
            if flag:
                break
            else:
                time.sleep(1)

    """检测器及信号灯设置API"""

    # 设置信号控制机(控制机<->交叉口)
    def set_signal_controller(self):
        self.sc = []
        for i in range(self.controllers.Count):
            controller = self.controllers.GetSignalControllerByNumber(i + 1)
            self.sc.append(controller)

    # 设置信号灯组(灯组<->相位)
    def set_signal_group(self):
        self.sg = []
        for i in range(self.groups.Count):
            group = self.groups.GetSignalGroupByNumber(i + 1)
            self.sg.append(group)

    # 设置行程时间检测器（检测器<->交叉口进出的一条完整道路）
    def set_travel_times_detector(self):
        self.tt, self.travel_time = [], []
        for i in range(self.travel_times.Count):
            travel_time = self.travel_times.GetTravelTimeByNumber(i + 1)
            self.tt.append(travel_time)
            self.travel_time.append(0)

    # 设置延误时间检测器（检测器<->交叉口平均延误，即多条道路的平均延误）
    def set_delay_times_detector(self):
        self.dt, self.delay = [], []
        for i in range(self.delays.Count):
            delay = self.delays.GetDelayByNumber(i + 1)
            self.dt.append(delay)
            self.delay.append(0)

    # 设置数据采集检测器（检测器<->进口道停车线）
    def set_data_collections_detector(self):
        self.dc, self.vel, self.speed = [], [], []
        for i in range(self.data_collections.Count):
            data_collection = self.data_collections.GetDataCollectionByNumber(i + 1)
            self.dc.append(data_collection)
            self.vel.append(0)
            self.speed.append(0)

    # 设置排队长度检测器（检测器<->进口道停车线）
    def set_queue_counters_detector(self):
        self.qc, self.queue_length = [], []  # 存储队列检测器对象和对应的队列长度
        for i in range(self.queue_counters.Count):
            queue_counter = self.queue_counters.GetQueueCounterByNumber(i + 1)  # 获取当前队列检测器对象
            self.qc.append(queue_counter)
            self.queue_length.append(0)  # 对应的每个队列检测器的长度都初始化为0

    # 设置仿真输入车流量--根据传入的流量列表，逐个设置每个车辆输入点的流量属性，并将输入点对象和流量值存储起来
    def set_vehicle_input_flow(self, flow):
        self.ip, self.ip_flow = [], []  # 存储车辆输入点对象和流量值

        # 循环遍历所有的车辆输入点（inputs）
        for i in range(self.inputs.Count):
            ip = self.inputs.GetVehicleInputByNumber(i + 1)  # 获取第 i+1 个车辆输入点对象
            ip.SetAttValue('VOLUME', flow[i])  # 将第 i 个车辆输入点的流量设置为 flow[i]
            self.ip.append(ip)
            self.ip_flow.append(flow[i])

    """信号灯控制API"""

    # 控制信号灯组
    def control_signal_group(self, plan):
        amber_time = plan[1]
        clearing_time = plan[2]
        green_time = plan[3]
        phase_num = len(green_time)
        value = [0 for _ in range(phase_num * 2)]

        for i in range(phase_num):
            if i == 0:
                value[i * 2] = 1  # 第一个相位（i=0），红灯结束时间设置为1，表示从仿真开始就是红灯
            else:
                value[i * 2] = value[i * 2 - 1] + amber_time + clearing_time[i - 1]  # 对于其他相位（i>0），红灯结束时间通过前一个相位的红灯结束时间加上黄灯时间和清理时间得到

            value[i * 2 + 1] = value[i * 2] + green_time[i]  # 绿灯结束时间则是红灯结束时间再加上当前相位的绿灯时间

            # sg数组：存储所有信号组对象，每个信号组控制着交叉口中的一个方向的车辆流向
            self.sg[i].SetAttValue("REDEND", value[i * 2])
            self.sg[i].SetAttValue("GREENEND", value[i * 2 + 1])

    """检测器数据提取API"""

    # 获取车流量采集检测器数据
    def get_flow_collections_detector(self):
        self.vel = []
        for i in range(len(self.dc)):
            self.vel.append(self.dc[i].GetResult("NVEHICLES", "sum", 0))
        return self.vel

    # 获取平均车速采集检测器数据
    def get_speed_collections_detector(self):
        self.speed = []
        for i in range(len(self.dc)):
            self.speed.append(self.dc[i].GetResult("SPEED", "mean", 0))
        return self.speed

    # 获取排队长度检测器数据
    def get_queue_counters_detector(self):
        self.queue_length = []
        elapsed_time = self.simulation.AttValue("ELAPSEDTIME")
        for i in range(len(self.qc)):
            self.queue_length.append(self.qc[i].GetResult(elapsed_time, "mean"))
        return self.queue_length

    # 获取延误检测器数据
    def get_delay_times_detector(self):
        self.delay = []
        elapsed_time = self.simulation.AttValue("ELAPSEDTIME")
        for i in range(len(self.dt)):
            self.delay.append(self.dt[i].GetResult(elapsed_time, "DELAY", "", 0))
        return self.delay
