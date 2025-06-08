from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUiType

ui, _ = loadUiType("./resource/ui/version.ui")


# 配置版本信息界面
class versionDialog(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowIcon(QIcon("../resource/icon/about.png"))
        self.setWindowTitle("版本信息")
        self.setFixedSize(self.width(), self.height())
