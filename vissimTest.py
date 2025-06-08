# 0.导入库（具体这个库参考其他资料）
import win32com.client  # 主要库
import time  # 不重要

# 1.连接VISSIM并创建VISSIM对象
vissim = win32com.client.gencache.EnsureDispatch("Vissim.Vissim.430")  # 最后数字为版本号
print(vissim)

# 2.加载路网(我们在绘制路网的时候通常会导入一张背景图，然后在上面绘制，注意这里仅加载路网不加载背景图)
filename = r"D:\Vissim4.3\Example\Test\test.inp"  # 好像不能用相对路径
vissim.LoadNet(filename)

# 3.仿真参数设置

# 仿真时长
period = vissim.Simulation.Period  # 读取仿真时长
print("仿真时长：", period)
period = vissim.Simulation.AttValue('Period')  # 另一种方式读取
print("仿真时长：", period)
vissim.Simulation.Period = 3600  # 更改仿真时长为3600s
print("仿真时长：", vissim.Simulation.Period)
vissim.Simulation.SetAttValue('Period', 1200)  # 另一种方法更改
print("仿真时长：", vissim.Simulation.Period)

# 随机种子
randomSeed = vissim.Simulation.RandomSeed  # 读取随机种子
print("随机种子：", randomSeed)
vissim.Simulation.SetAttValue('RandomSeed', 35)  # 更改随机种子
print("仿真时长：", vissim.Simulation.RandomSeed)

# 仿真形式
# #仿真连续运行到结束
# vissim.Simulation.RunContinuous()
# 仿真单步运行
for i in range(1, 6010):  # 仿真时长
    print("当前仿真时刻：", i)
    # 4.获取车辆信息

    # 对总车辆的操作
    vehicles = vissim.Net.Vehicles  # 当前仿真时刻整个道路上的车辆集合
    vehicles_count = vehicles.Count  # 当前仿真时刻整个道路上的车辆总数
    print("车辆总数：", vehicles_count)
    vehicles_id = vehicles.IDs
    print("车辆编号：", vehicles_id)  # 当前仿真时刻整个道路上的车辆编号集合

    vissim.Simulation.RunSingleStep()  # 一步一步仿真直到仿真时间结束


# 停止仿真
vissim.Simulation.Stop()
