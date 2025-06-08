import os
import time
import psutil


# 检测vissim是否运行
def detect_vissim():
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
def run_vissim_exe():
    # 检测vissim是否运行
    flag = detect_vissim()
    if flag:
        # 强制结束当前vissim进程
        os.system("taskkill /F /IM vissim.exe")
    # 自动重启vissim软件
    os.system(r'RunAsDate.exe /movetime 03\10\2008 15:33:00 "%s"' % "D:\\Vissim4.3\\Exe\\vissim.exe")

    while True:
        # 等待重启成功
        flag = detect_vissim()
        if flag:
            print("Vissim成功启动！")
            break
        else:
            time.sleep(1)


run_vissim_exe()
