import os
import sys

from PyQt5.QtCore import QCoreApplication, QTranslator, Qt
from PyQt5.QtWidgets import QApplication
from interface.window import MainWindow


# 检查文件路径
def check_path():
    path_list = ['./model', './model/pkl', './model/png', './model/txt', './backup']
    for path in path_list:
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)


# CPU电源模式配置
def power_config(index=2):
    import subprocess
    mode = [
        "a1841308-3541-4fab-bc81-f71556f20b4a",  # 节能
        "381b4222-f694-41f0-9685-ff5bb260df2e",  # 平衡
        "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",  # 高性能
        "e9a42b02-d5df-448d-aa00-03f14749eb61",  # 卓越
    ]

    # 执行 Windows 系统命令，用于设置电源配置选项
    subprocess.call("Powercfg -s %s" % mode[index])


if __name__ == '__main__':
    power_config()
    check_path()

    # 确保应用程序的界面元素能够正确地显示和缩放
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    trans = QTranslator()
    trans.load("zh_CN")
    app.installTranslator(trans)

    window = MainWindow()
    window.show()

    # 启动 Qt 应用程序的事件循环
    sys.exit(app.exec_())
