# DRL-Dynamic-Timing
This is a repository about a creative project to finish the dynamic timing of traffic light.

## 环境

torch 2.3.1+cu121 + PyQt 5.15.4 + numpy 1.24.3

## 参考文献

```
李珊.道路交叉口交通监测及信号灯配时研究与设计[D].西安工业大学,2022.DOI:10.27391/d.cnki.gxagu.2022.000221.
```

## 流程

设计了agent智能体神经网络

env环境采用VISSIM模拟真实交通情况，通过Python COM接口建立连接

在当前状态交通数据下，agent采用ε-greedy策略选择action

在此action下仿真模拟一周期，返回下一状态交通数据

利用存储在经验池中的元组进行训练，训练过程即为DQN迭代

使得agent总是根据当前state选择最佳action，达到动态配时效果

PyQt设计训练界面，展示图表结果，对原论文界面进行简化

## 使用

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xxx
```

## VISSIM说明

采用的是VISSIM 4.3版本，由于版本问题，设置了startVissim.py脚本，直接运行后自动打开VISISM。否则手动打开需设置时间2008年。
