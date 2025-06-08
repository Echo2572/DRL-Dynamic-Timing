import ctypes
import os
import time
from shutil import move

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pythoncom
import win32con

from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QTextCursor, QFont
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QMessageBox, QGridLayout
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from win32process import SuspendThread, ResumeThread

from interface import version, contact
from simulation import simulation

ui, _ = loadUiType("./resource/ui/window.ui")
demo = False


# 仿真运行在单独的线程中可以避免阻塞主线程，保持界面的响应性
class MyThread(QThread):
    handle = -1
    stop_flag = False
    save_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.sim = simulation.Sim()
        self.para = []

    # 重写run方法，运行一个完整的仿真过程
    def run(self):
        try:
            # 获取当前线程的句柄
            self.handle = ctypes.windll.kernel32.OpenThread(win32con.PROCESS_ALL_ACCESS, False,
                                                            int(QThread.currentThreadId()))
        except (IndexError, Exception):
            pass

        if not self.stop_flag:
            # 初始化资源
            pythoncom.CoInitialize()
            self.sim.run(self.para[0], self.para[1], self.para[2])  # 进行仿真运算
            print("仿真结束")
            self.save_signal.emit(True)  # 发送信号 save_signal 表明仿真执行完成
        else:
            pass


# 自定义画布类，用于绘图
class PlotCanvas(FigureCanvas):
    # 画布类的初始化操作
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        # figsize 参数表示绘图区域的尺寸，以英寸为单位；dpi 参数表示绘图区域的分辨率，即每英寸像素数
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        font = {'family': 'SimSun', 'size': '8'}  # SimSun是一种中文字体

        # 设置了一些绘图相关的默认参数
        plt.rc('font', **font)
        plt.rc('axes', unicode_minus=False)
        plt.rcParams['figure.facecolor'] = "#FFFFF0"  # 设置窗体颜色
        plt.rcParams['axes.facecolor'] = "#FFFFF0"  # 设置绘图区颜色
        plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体样式
        plt.rcParams['font.size'] = '8'  # 设置字体大小

        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)  # 更新绘图区部件的几何信息

    # 根据是否为demo选取训练文件
    def verify_path(self):
        if demo:
            self.train_drl_file = './test/model/txt/train_record.txt'
            self.train_test_drl_file = './test/model/txt/test_record.txt'
            self.test_drl_file = "./test/model/txt/test_drl_record.txt"
            self.test_fix_file = "./test/fix/txt/test_fix_record.txt"
        else:
            self.train_drl_file = './model/txt/train_record.txt'
            self.train_test_drl_file = './model/txt/test_record.txt'
            self.test_drl_file = "./model/txt/test_drl_record.txt"
            self.test_fix_file = "./fix/txt/test_fix_record.txt"

    # 清理画布
    def clear_fig(self):
        self.fig.clf()  # 清理画布
        self.fig.canvas.draw()  # 重绘画布
        self.fig.canvas.flush_events()  # 更新画布

    # 评估界面第一部分画布--深度强化配时仿真训练结果，根据传入的pic数量画图
    def plot_drl_train(self, pic):
        self.verify_path()

        # 读取drl训练数据
        names = ["episode", "step", "epsilon", "learn_rate", "convergence", "delay", "reward", "loss"]
        data = pd.read_csv(self.train_drl_file, sep="\s+", names=names)

        # 将数据存储在相应的列表中
        delay = list(data["delay"].values)
        reward = list(data["reward"].values)
        loss = list(data["loss"].values)
        convergence = list(data["convergence"].values)
        epsilon = list(data["epsilon"].values)
        learn_rate = list(data["learn_rate"].values)

        # 返回一个包含非零元素的子列表
        idx = 0
        for i in range(len(loss)):
            if loss[i] != 0:
                idx = i
                break
        loss = loss[idx::]

        # 读取drl训练测试数据
        names = ["episode", "step", "delay", "reward"]
        data = pd.read_csv(self.train_test_drl_file, sep="\s+", names=names)

        # 数据存储为列表
        train_test_delay = list(data["delay"].values)
        train_test_reward = list(data["reward"].values)

        y = [delay, reward, loss, epsilon, learn_rate, convergence, train_test_delay, train_test_reward]
        pic = np.array(pic)  # 将列表转换为数组
        num = len(pic[pic != 0])  # 统计 pic 数组中非零元素的个数

        n = 0  # 当前绘制的子图数量
        self.fig.clf()  # 清理画布
        for i in range(len(pic)):
            if pic[i] != 0:
                n += 1
                ax = self.fig.add_subplot(1, num, n)  # 将画布分成 1 行、num 列的网格，并在第 n 个位置上绘制子图
                x = np.linspace(0, len(y[i]), len(y[i]))  # 根据起始值、结束值和数组长度信息，在这个范围内生成指定数量的等间距的数值
                ax.plot(x, y[i], "k")  # 在当前子图上绘制图像数据

        self.fig.canvas.draw()  # 重绘画布
        self.fig.canvas.flush_events()  # 更新画布

    # 评估界面第二部分画布--深度强化配时仿真测试结果，根据传入的pic数量画图，且每个pic图像类型不同
    def plot_drl_test(self, pic):
        self.verify_path()

        # 读取drl配时测试数据
        names = ["step", "plan", "delay", "reward"]
        data = pd.read_csv(self.test_drl_file, sep="\s+", names=names)

        # drl延误
        drl_delay = list(data["delay"].values)
        drl_plan = list(data["plan"].values)

        # drl平均延误
        mean_delay = round(sum(drl_delay) / len(drl_delay), 3)  # 四舍五入至3位小数

        # 读取固定配时测试数据
        names = ["plan", "step", "delay"]
        data = pd.read_csv(self.test_fix_file, sep="\s+", names=names)
        fix_plan = list(data["plan"].values)
        fix_delay = list(data["delay"].values)
        compar = list(map(lambda x: round((x - mean_delay) / x * 100, 2), fix_delay))

        pic = np.array(pic)
        num = len(pic[pic != 0])

        n = 0
        self.fig.clf()  # 清理画布
        for i in range(len(pic)):
            if pic[i] != 0:
                n += 1
                ax = self.fig.add_subplot(1, num, n)

                if i == 0:
                    x = np.linspace(0, len(drl_delay) - 1, len(drl_delay))
                    ax.plot(x, drl_delay, color='black', marker='D', linestyle='-', linewidth='1.0')
                    ax.plot(x, [mean_delay for _ in range(len(x))], color='gray', linestyle='--')
                elif i == 1:
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
                    ax.bar(x, plan_freq, 0.5, color="gray", edgecolor="k")
                elif i == 2:
                    x = np.linspace(0, len(compar) - 1, len(compar))
                    ax.plot(x, fix_delay, color='black', marker='D', linestyle='-')
                    ax.plot(x, [mean_delay for _ in range(len(fix_plan))], color='gray', marker='*', linestyle='--')
                elif i == 3:
                    # 指定固定配时方案延误
                    specify_plan = pic[i]
                    specify_fix_file = self.test_fix_file[:-4] + '_' + str(specify_plan) + '.txt'

                    names = ["step", "delay"]
                    data = pd.read_csv(specify_fix_file, sep="\t+", names=names, engine="python")
                    fix_specify_delay = list(data["delay"].values)
                    fix_specify_mean_delay = round(sum(fix_specify_delay) / len(fix_specify_delay), 3)

                    x = np.linspace(0, len(fix_specify_delay) - 1, len(fix_specify_delay))
                    ax.plot(x, fix_specify_delay, color='gray', marker='o', linestyle=':')
                    ax.plot(x, drl_delay, color='black', marker='D', linestyle='--')
                    ax.plot(x, [fix_specify_mean_delay for _ in range(len(fix_specify_delay))], color='gray',
                            marker='o', linewidth=0.5, linestyle='--')
                    ax.plot(x, [mean_delay for _ in range(len(drl_delay))], color='black', marker='D', linewidth=0.5,
                            linestyle='--')

        self.fig.canvas.draw()  # 重绘画布
        self.fig.canvas.flush_events()  # 更新画布

    # 评估界面第三部分画布--固定配时仿真测试结果，根据传入的pic数量画图，且每个pic图像类型不同
    def plot_fix_test(self, pic):
        self.verify_path()

        # 读取固定配时测试数据
        names = ["plan", "step", "delay"]
        data = pd.read_csv(self.test_fix_file, sep="\s+", names=names)
        fix_plan = list(data["plan"].values)
        fix_delay = list(data["delay"].values)

        min_delay = min(filter(lambda x: x > 0, fix_delay))
        fix_min_idx = fix_plan[fix_delay.index(min_delay)]  # 最小延迟对应的计划
        fix_max_idx = fix_plan[fix_delay.index(max(fix_delay))]  # 最大延迟对应的计划

        # min-fix 延误
        min_fix_file = self.test_fix_file[:-4] + '_' + str(fix_min_idx) + '.txt'
        names = ["step", "delay"]
        data = pd.read_csv(min_fix_file, sep="\t+", names=names, engine="python")
        fix_min_delay = list(data["delay"].values)

        pic = np.array(pic)
        num = len(pic[pic != 0])

        n = 0
        self.fig.clf()  # 清理画布
        for i in range(len(pic)):
            if pic[i] != 0:
                n += 1
                ax = self.fig.add_subplot(1, num, n)
                if i == 0:
                    x = np.linspace(1, len(fix_delay), len(fix_delay))
                    ax.bar(fix_min_idx, min_delay, 0.5, label='min delay plan', color="w", edgecolor="k",
                           hatch="\\\\\\\\")
                    ax.bar(fix_max_idx, max(fix_delay), 0.5, label='max delay plan', color="w", edgecolor="k",
                           hatch="/////")
                    ax.plot(x, [np.mean(fix_delay) for _ in range(len(x))], label='mean delay line', color='gray',
                            linestyle='--')
                elif i == 1:
                    x = np.linspace(1, len(fix_min_delay), len(fix_min_delay))
                    mean_delay = round(sum(fix_min_delay) / len(fix_min_delay), 3)
                    ax.plot(x, fix_min_delay, label='min delay', color='black', linestyle='--')
                    ax.plot(x, [mean_delay for _ in range(len(fix_min_delay))], color='black', linestyle='--')
                elif i == 2:
                    # 指定固定配时方案延误
                    specify_plan = pic[i]
                    specify_fix_file = self.test_fix_file[:-4] + '_' + str(specify_plan) + '.txt'

                    names = ["step", "delay"]
                    data = pd.read_csv(specify_fix_file, sep="\t+", names=names, engine="python")
                    fix_specify_delay = list(data["delay"].values)
                    mean_delay = round(sum(fix_specify_delay) / len(fix_specify_delay), 3)

                    x = np.linspace(0, len(fix_specify_delay) - 1, len(fix_specify_delay))
                    ax.plot(x, fix_specify_delay, color='black', marker='o', linestyle=':')
                    ax.plot(x, [mean_delay for _ in range(len(fix_specify_delay))], color='black', linestyle='--')

        self.fig.canvas.draw()  # 重绘画布
        self.fig.canvas.flush_events()  # 更新画布


# 主窗口界面
class MainWindow(QMainWindow, ui):
    def __init__(self):
        # 窗口属性
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.resize(1366, 700)
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setWindowTitle("基于DQN算法的红绿灯配时系统")
        self.setWindowIcon(QIcon("./resource/icon/app.png"))
        self.tabWidget.tabBar().setVisible(False)  # 隐藏Tab标签栏

        # 类属性
        self.net_path = "E:\\Vissim-Python-Qt\\resource\\vissim\\net\\net.inp"
        self.simulation = [999999, 0, 1, 1, 42, True, False]  # 仿真设置：最大回合仿真时长，仿真速度，模拟分辨率，控制频率，图形模式，3D模式
        self.timing = [170, 41, 57, 2, 3, [2, 2, 2]]  # 信号配时设置：信号周期，最短绿时，最长绿时，绿时变化间隔，黄灯时间，各相位全红时间
        self.plans = []
        self.pic1 = [0 for _ in range(8)]  # 生成一个由8个0组成的列表
        self.pic2 = [0 for _ in range(6)]
        self.pic3 = [0 for _ in range(4)]

        # 实例化其他接口类
        self.versionDialog = version.versionDialog()
        self.contactDialog = contact.contactDialog()

        # 设置图标
        self.label_net.setPixmap(QPixmap("./resource/icon/net.png"))
        self.label_alg.setPixmap(QPixmap("./resource/icon/algorithm.png"))
        self.label_sim.setPixmap(QPixmap("./resource/icon/simulation.png"))
        self.label_eval.setPixmap(QPixmap("./resource/icon/evaluate.png"))

        # 让图片自适应label大小
        self.label_net.setScaledContents(True)
        self.label_alg.setScaledContents(True)
        self.label_sim.setScaledContents(True)
        self.label_eval.setScaledContents(True)

        # 主界面切换槽函数连接
        self.pushButton1.clicked.connect(self.switch_to_net_panel)
        self.pushButton2.clicked.connect(self.switch_to_algorithm_panel)
        self.pushButton3.clicked.connect(self.switch_to_simulation_panel)
        self.pushButton4.clicked.connect(self.switch_to_evaluate_panel)

        # 菜单栏触发槽函数连接
        self.action_load.triggered.connect(self.load_net_file)
        self.action_exit.triggered.connect(self.quit_program)
        self.action_local.triggered.connect(self.open_help_document)
        self.action_version.triggered.connect(self.versionDialog.show)
        self.action_contact.triggered.connect(self.contactDialog.show)

        # 路网界面槽函数连接
        self.pushButton_load.clicked.connect(self.load_net_file)

        # 仿真界面槽函数连接
        self.pushButton_simulation_run.clicked.connect(self.run_simulation)
        self.pushButton_simulation_stop_continue.clicked.connect(self.stop_continue_simulation)
        self.pushButton_simulation_quit.clicked.connect(self.quit_simulation)

        # 评估界面-深度强化训练-槽函数连接
        self.checkBox_train_delay.toggled.connect(self.select_train_drl_display)
        self.checkBox_train_reward.toggled.connect(self.select_train_drl_display)
        self.checkBox_train_loss.toggled.connect(self.select_train_drl_display)
        self.checkBox_train_epsilon.toggled.connect(self.select_train_drl_display)
        self.checkBox_train_lr.toggled.connect(self.select_train_drl_display)
        self.checkBox_train_test_delay.toggled.connect(self.select_train_drl_display)
        self.checkBox_train_test_reward.toggled.connect(self.select_train_drl_display)
        self.checkBox_train_conv.toggled.connect(self.select_train_drl_display)
        self.checkBox_demo.toggled.connect(self.switch_demo_model)

        # 评估界面-深度强化测试-槽函数连接
        self.checkBox_test_delay.toggled.connect(self.select_test_drl_display)
        self.checkBox_plan_freq.toggled.connect(self.select_test_drl_display)
        self.checkBox_compare_all_fix.toggled.connect(self.select_test_drl_display)
        self.checkBox_compare_specify_fix.toggled.connect(self.select_test_drl_display)
        self.spinBox_plan.valueChanged.connect(self.select_test_drl_display)

        # 评估界面-固定配时测试-槽函数连接
        self.checkBox_fix_all_test_delay.toggled.connect(self.select_test_fix_display)
        self.checkBox_fix_min_test_delay.toggled.connect(self.select_test_fix_display)
        self.checkBox_specify_fix.toggled.connect(self.select_test_fix_display)
        self.spinBox_plan_fix.valueChanged.connect(self.select_test_fix_display)

        # QSS样式表，切换两种不同类型的按钮(仿真进行中还是停止)
        self.QSS_sheet = [
            "QPushButton{border:none;color:black;font-size:11;border-radius:10px;padding-left:5px;padding-right:10px;\
            text-align:middle;background:LightGray;background-color:#FF69B4;}"
            "QPushButton:hover{color:white;border:1px solid #F3F3F5;border-radius:10px;background:#00FF00;}"
            "QPushButton:pressed{color:red;border:3px solid #4169E1;border-radius:10px;background:#00FF00;}",
            "QPushButton{border:none;color:black;font-size:11;border-radius:10px;padding-left:5px;padding-right:10px;\
            text-align:middle;background:LightGray;background-color:#808080;}"
            "QPushButton:hover{color:white;border:1px solid #F3F3F5;border-radius:10px;background:#00FF00;}"
            "QPushButton:pressed{color:red;border:3px solid #4169E1;border-radius:10px;background:#00FF00;}"
        ]

        # 初始化按钮默认状态(刚开始停止仿真和退出仿真按钮不可用)
        self.pushButton_simulation_quit.setEnabled(False)
        self.pushButton_simulation_quit.setStyleSheet(self.QSS_sheet[1])
        self.pushButton_simulation_stop_continue.setEnabled(False)
        self.pushButton_simulation_stop_continue.setStyleSheet(self.QSS_sheet[1])

        # 注册子线程服务,初始化一个自定义的线程对象
        self.sim_task = MyThread()
        self.sim = self.sim_task.sim
        self.sim_task.finished.connect(self.sim_task.deleteLater)

        # 连接自定义信号
        self.sim.SimInfoEmit.connect(self.update_simulation_textBrowser)
        self.sim.RateProgressInfoEmit.connect(self.update_simulation_progressBar)
        self.sim.RemainTimeInfoEmit.connect(self.update_simulation_time)

        # 更新系统信息的定时器
        self.system_state = QTimer()
        self.system_state.timeout.connect(self.update_system_state)
        self.system_state.stop()

        # 更新可视化图形的定时器
        self.visual_evaluation = QTimer()
        self.visual_evaluation.timeout.connect(self.update_figure)
        self.visual_evaluation.stop()

        # 绘图区域，展示图形数据
        self.F = PlotCanvas()
        self.F1 = PlotCanvas(width=4, height=5, dpi=30)
        self.F2 = PlotCanvas(width=4, height=5, dpi=30)
        self.F3 = PlotCanvas(width=4, height=5, dpi=30)

        # 创建网格布局管理器
        self.gridlayout_1 = QGridLayout(self.groupBox_plot_1)
        self.gridlayout_2 = QGridLayout(self.groupBox_plot_2)
        self.gridlayout_3 = QGridLayout(self.groupBox_plot_3)

        # 绘图区域和布局管理器结合
        self.gridlayout_1.addWidget(self.F1, 0, 1)
        self.gridlayout_2.addWidget(self.F2, 0, 1)
        self.gridlayout_3.addWidget(self.F3, 0, 1)

        # 初始进度条信息
        self.progressBar_memory.setValue(0)
        self.progressBar_cpu.setValue(0)
        self.progressBar_gpu.setValue(0)
        self.progressBar_simulation.setValue(0)

        self.update_system_state()  # 显示系统状态信息
        self.switch_to_net_panel()  # 默认显示路网界面

    # 切换demo模式
    def switch_demo_model(self):
        global demo
        if self.checkBox_demo.isChecked():
            demo = True
        else:
            demo = False

        # 清空画布，重新画图
        self.F1.clear_fig()
        self.F2.clear_fig()
        self.F3.clear_fig()
        self.update_figure()

    # 备份网络模型
    @staticmethod
    def backup_model(source, target):
        # 遍历源目录下的所有文件和子目录
        for root, dirs, files in os.walk(source):
            if len(files) > 0:
                now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))  # 获取系统时间

                path = target + now + '/'  # 创建新备份子文件夹
                if os.path.exists(path):
                    pass
                else:
                    os.makedirs(path)

                # 移动文件
                for file in files:
                    move(os.path.join(root, file), path)

    # 创建配时方案
    @staticmethod
    def create_plans(para, file, plan_num=-1):
        # 配时范围
        cycle_time = para[0]  # 周期时间--完整的变化周期
        green_low = para[1]  # 最短绿灯时间
        green_high = para[2]  # 最长绿灯时间
        green_interval = para[3]  # 绿灯间隔时间
        amber_time = para[4]  # 黄灯时间
        clearing_time = para[5]  # 清场时间--每个信号灯切换后确保所有方向的车辆都通过交叉口
        phase_num = len(clearing_time)
        loss_time = amber_time * phase_num + sum(clearing_time)

        try:
            with open(file, "a+") as f:
                f.truncate(0)  # 将文件对象 f 的内容截断为0字节
        except (IndexError, Exception):
            pass

        from itertools import permutations
        a = list(permutations(range(green_low, green_high + 1, green_interval), phase_num))
        plans = []
        unused_plan = 0

        for i in range(len(a)):
            if sum(a[i]) == cycle_time - loss_time:
                if plan_num < 0 or len(plans) < plan_num:
                    plan = [cycle_time, amber_time, clearing_time, list(a[i])]
                    plans.append(plan)

                    with open(file, "a+") as f:
                        line = "%s\n" % (str(plan))
                        f.write(line)
                else:
                    unused_plan += 1
        return plans, unused_plan

    # 显示系统状态信息
    def update_system_state(self):
        # 内存占用
        mem = psutil.virtual_memory()
        mem_percent = float(mem.percent)
        self.progressBar_memory.setValue(mem_percent)
        self.progressBar_memory.update()

        # CPU占用
        cpu_percent = psutil.cpu_percent(1)
        self.progressBar_cpu.setValue(cpu_percent)
        self.progressBar_cpu.update()

        # GPU占用
        import GPUtil
        GPUs = GPUtil.getGPUs()
        gpu_load = GPUs[0].load
        gpu_percent = gpu_load * 100
        self.progressBar_gpu.setValue(gpu_percent)
        self.progressBar_gpu.update()

    # 退出程序
    def quit_program(self):
        reply = QMessageBox.question(self, '提示', "是否确认退出仿真系统？", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()

    # 打开帮助文档
    def open_help_document(self):
        path = os.getcwd() + r"/README.txt"
        if os.path.exists(path):
            os.startfile(path)
        else:
            QMessageBox.warning(self, "提示", "当前目录下帮助文档不存在")

    # 切换到路网界面
    def switch_to_net_panel(self):
        self.tabWidget.setCurrentIndex(0)
        self.statusBar().showMessage("路网界面")
        self.system_state.stop()
        self.visual_evaluation.stop()

    # 切换到算法界面
    def switch_to_algorithm_panel(self):
        self.tabWidget.setCurrentIndex(1)
        self.statusBar().showMessage("算法界面")
        self.system_state.stop()
        self.visual_evaluation.stop()

    # 切换到仿真界面·
    def switch_to_simulation_panel(self):
        self.tabWidget.setCurrentIndex(2)
        self.statusBar().showMessage("仿真界面")
        self.update_system_state()  # 显示系统状态
        self.visual_evaluation.stop()
        self.system_state.start(10 * 1000)  # 启动系统状态，每隔 10 秒更新一次

    # 切换到评价界面
    def switch_to_evaluate_panel(self):
        self.tabWidget.setCurrentIndex(3)
        self.statusBar().showMessage("评估界面")
        self.switch_demo_model()

        # 可视化展示图像
        self.select_train_drl_display()
        self.select_test_drl_display()
        self.select_test_fix_display()

        self.system_state.stop()
        self.visual_evaluation.start(10 * 1000)

    """路网界面"""

    # 加载路网文件
    def load_net_file(self):
        reply = QFileDialog.getOpenFileName(self, "道路仿真路网文件", "./resource/", "(*.inp)")
        try:
            if reply[0]:
                self.net_path = reply[0]
                try:
                    with open(self.net_path, "r") as f:
                        content = f.readlines()

                    self.textBrowser_net_info.clear()  # 文本区域清空
                    for i in range(len(content)):
                        self.textBrowser_net_info.append(content[i])
                    self.textBrowser_net_info.moveCursor(QTextCursor.Start)  # 将光标移动到文本的开头

                except (IndexError, Exception):
                    QMessageBox.warning(self, "提示", "加载路网文件失败")
                QMessageBox.information(self, "提示", "加载路网文件成功")

        except (IndexError, Exception):
            QMessageBox.warning(self, "提示", "加载路网文件失败")

    """算法界面"""

    # 加载仿真训练参数设置
    def load_train_setting(self):
        simulation.EPISODE = int(self.lineEdit_EPISODE.text())
        simulation.STEP = int(self.lineEdit_STEP.text())
        simulation.WARM_STEP = int(self.lineEdit_WARM_STEP.text())
        simulation.TEST_EPISODE = float(self.lineEdit_TEST_EPISODE.text())
        simulation.CONVERGENCE_LOW = int(self.lineEdit_CONVERGENCE_LOW.text())
        simulation.CONVERGENCE_UP = int(self.lineEdit_CONVERGENCE_UP.text())
        simulation.ZEROREWARD = int(self.lineEdit_ZEROREWARD.text())
        simulation.CONVERGENCE = int(self.lineEdit_CONVERGENCE.text())

    # 加载仿真算法参数设置
    def load_algorithm_setting(self):
        simulation.LR_MAX = float(self.lineEdit_LR_MAX.text())
        simulation.LR_MIN = float(self.lineEdit_LR_MIN.text())
        simulation.EPSILON_MAX = float(self.lineEdit_EPSILON_MAX.text())
        simulation.EPSILON_MIN = float(self.lineEdit_EPSILON_MIN.text())
        simulation.MEMORY_CAPACITY = int(self.lineEdit_MEMORY_CAPACITY.text())
        simulation.BATCH_SIZE = int(self.lineEdit_BATCH_SIZE.text())
        simulation.UPDATE_STEP = int(self.lineEdit_UPDATE_STEP.text())
        simulation.GAMMA = float(self.lineEdit_GAMMA.text())

    """仿真界面"""

    # 显示提示信息
    def show_hint_information(self):
        self.textBrowser_sim_info.clear()
        self.textBrowser_sim_info.append("Real-time simulation information:\n")

    # 刷新仿真信息文本浏览框
    def update_simulation_textBrowser(self, info):
        self.textBrowser_sim_info.append(info)
        QApplication.processEvents()

    # 刷新仿真进度条
    def update_simulation_progressBar(self, progress):
        simulation_percent = progress
        self.progressBar_simulation.setValue(simulation_percent)
        self.progressBar_simulation.update()

    # 刷新仿真剩余时间
    def update_simulation_time(self, h, m, s):
        info = '剩余时间：%02d时%02d分%02d秒' % (h, m, s)
        self.label_remain_time.setText(info)
        self.label_remain_time.setFont(QFont("SimSun-ExtB", 11))

    # 检测vissim是否运行
    @staticmethod
    def detect_vissim():
        try:
            pids = psutil.pids()  # 获取当前系统中所有进程的 PID
            for pid in pids:
                p = psutil.Process(pid)
                process_name = p.name()

                if "vissim.exe" in process_name:
                    return True
            else:
                return False

        except:
            return False

    # 运行仿真
    def run_simulation(self):
        # 检测vissim是否运行
        start_flag = self.detect_vissim()
        if start_flag:
            if simulation.device == "cuda":
                reply = QMessageBox.information(self, "提示", "系统检测到GPU,是否选择GPU进行训练",
                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.No:
                    simulation.device = "cpu"

            # 更改按钮状态
            self.pushButton_simulation_run.setEnabled(False)
            self.pushButton_simulation_stop_continue.setEnabled(True)
            self.pushButton_simulation_quit.setEnabled(True)

            # 更改按钮样式
            self.pushButton_simulation_run.setStyleSheet(self.QSS_sheet[1])
            self.pushButton_simulation_stop_continue.setStyleSheet(self.QSS_sheet[0])
            self.pushButton_simulation_quit.setStyleSheet(self.QSS_sheet[0])

            self.backup_model('./model/', './backup/')  # 备份已存在的网络模型文件
            self.show_hint_information()  # 显示提示信息

            self.load_algorithm_setting()  # 加载仿真算法参数
            self.load_train_setting()  # 加载仿真训练参数

            # 生成配时方案
            plans_file = "./model/txt/test_fix_plans.txt"
            self.plans, _ = self.create_plans(self.timing, plans_file, 20)

            # 子线程启动仿真(子线程中调用的是simulation中的run方法)
            self.sim_task.stop_flag = False
            self.sim_task.para = [self.net_path, self.simulation, self.plans]  # 仿真过程传递的实参
            self.sim_task.start()  # 线程对象会调用run方法

        else:
            QMessageBox.warning(self, "警告", "请先启动Vissim")

    # 停止/继续仿真
    def stop_continue_simulation(self):
        if self.pushButton_simulation_stop_continue.text() == "停止仿真":
            self.pushButton_simulation_stop_continue.setText("继续仿真")
            self.pushButton_simulation_quit.setEnabled(True)
            self.pushButton_simulation_quit.setStyleSheet(self.QSS_sheet[0])

            if self.sim_task.handle == -1:
                return 0

            # 暂停线程
            SuspendThread(self.sim_task.handle)
            info = "Stop of the Simulation"
            self.textBrowser_sim_info.append(info)

        elif self.pushButton_simulation_stop_continue.text() == "继续仿真":
            self.pushButton_simulation_stop_continue.setText("停止仿真")
            self.pushButton_simulation_quit.setEnabled(False)
            self.pushButton_simulation_quit.setStyleSheet(self.QSS_sheet[1])

            if self.sim_task.handle == -1:
                return 0

            # 恢复线程
            ResumeThread(self.sim_task.handle)
            info = "Continue of the Simulation"
            self.textBrowser_sim_info.append(info)

    # 结束仿真
    def quit_simulation(self):
        # 销毁子线程服务
        ctypes.windll.kernel32.TerminateThread(self.sim_task.handle, 0)

        self.sim_task.stop_flag = True
        self.sim_task.stop_flag = True

        self.pushButton_simulation_quit.setEnabled(False)
        self.pushButton_simulation_run.setEnabled(True)
        self.pushButton_simulation_stop_continue.setEnabled(False)
        self.pushButton_simulation_run.setStyleSheet(self.QSS_sheet[0])
        self.pushButton_simulation_quit.setStyleSheet(self.QSS_sheet[1])
        self.pushButton_simulation_stop_continue.setStyleSheet(self.QSS_sheet[1])
        self.pushButton_simulation_stop_continue.setText("停止仿真")

        # 重新注册线程
        self.sim_task = MyThread()
        self.sim_task.finished.connect(self.sim_task.deleteLater)

        # 连接自定义信号
        self.sim = self.sim_task.sim
        self.sim.SimInfoEmit.connect(self.update_simulation_textBrowser)

        # 结束仿真
        info = "End of the Simulation"
        self.textBrowser_sim_info.append(info)

    # 重写主窗口的关闭事件--加上删除子线程
    def closeEvent(self, event):
        if self.sim_task.isRunning():
            self.sim_task.quit()
            self.sim_task.terminate()

        del self.sim_task
        super(MainWindow, self).closeEvent(event)

    """评估信息界面"""

    # 刷新可视化界面
    def update_figure(self):
        try:
            self.F1.plot_drl_train(self.pic1)
            self.F2.plot_drl_test(self.pic2)
            self.F3.plot_fix_test(self.pic3)
        except:
            QMessageBox.warning(self, "警告", "读取数据文件异常")

    # 选择训练深度强化学习可视化展示图像
    def select_train_drl_display(self):
        self.pic1 = [0 for _ in range(8)]

        if self.checkBox_train_delay.isChecked():
            self.pic1[0] = 1
        if self.checkBox_train_reward.isChecked():
            self.pic1[1] = 1
        if self.checkBox_train_loss.isChecked():
            self.pic1[2] = 1
        if self.checkBox_train_epsilon.isChecked():
            self.pic1[3] = 1
        if self.checkBox_train_lr.isChecked():
            self.pic1[4] = 1
        if self.checkBox_train_conv.isChecked():
            self.pic1[5] = 1
        if self.checkBox_train_test_delay.isChecked():
            self.pic1[6] = 1
        if self.checkBox_train_test_reward.isChecked():
            self.pic1[7] = 1

        try:
            self.F1.plot_drl_train(self.pic1)
        except:
            print("F1错误")
            QMessageBox.warning(self, "警告", "读取深度强化配时训练数据文件异常")

    # 选择训练深度强化学习可视化展示图像
    def select_test_drl_display(self):
        self.pic2 = [0 for _ in range(4)]

        if self.checkBox_test_delay.isChecked():
            self.pic2[0] = 1
        if self.checkBox_plan_freq.isChecked():
            self.pic2[1] = 1
        if self.checkBox_compare_all_fix.isChecked():
            self.pic2[2] = 1
        if self.checkBox_compare_specify_fix.isChecked():
            plan = self.spinBox_plan.value()
            self.pic2[3] = plan

        try:
            self.F2.plot_drl_test(self.pic2)
        except:
            print("F2错误")
            QMessageBox.warning(self, "警告", "读取深度强化配时测试数据文件异常")

    # 选择训练深度强化学习可视化展示图像
    def select_test_fix_display(self):
        self.pic3 = [0 for _ in range(3)]

        if self.checkBox_fix_all_test_delay.isChecked():
            self.pic3[0] = 1
        if self.checkBox_fix_min_test_delay.isChecked():
            self.pic3[1] = 1
        if self.checkBox_specify_fix.isChecked():
            plan = self.spinBox_plan_fix.value()
            self.pic3[2] = plan

        try:
            self.F3.plot_fix_test(self.pic3)
        except:
            print("F3错误")
            QMessageBox.warning(self, "警告", "读取固定配时测试数据文件异常")
