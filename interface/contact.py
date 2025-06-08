from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUiType

ui, _ = loadUiType("./resource/ui/contact.ui")


# 联系与反馈信息界面
class contactDialog(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowIcon(QIcon("../resource/icon/about.png"))
        self.setWindowTitle("联系反馈")
        self.setFixedSize(self.width(), self.height())

        # 增强 QLabel 组件的交互性，使其更加友好和易用
        self.label.setOpenExternalLinks(True)  # 设置 QLabel 组件支持打开外部链接
        self.label.setTextInteractionFlags(Qt.TextBrowserInteraction)  # 设置 QLabel 组件支持文本浏览器交互
