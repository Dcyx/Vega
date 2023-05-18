import openai
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from threading import Thread


class ChatWindowBubbleRight(QWidget):
    def __init__(self, parent, **kwargs):
        QtWidgets.QWidget.__init__(self)
        # Window Config
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 无边框 + 窗口置顶
        self.setAttribute(Qt.WA_TranslucentBackground)  # 半透明背景
        self.setAutoFillBackground(True)  # 非自动填充
        self.repaint()

        # Window UI
        self.img = QLabel(self)
        self.set_pic("chat_bubble_right.png")
        self.resize(512, 512)

    def set_pic(self, pic):
        img = QImage()
        img.load('img/'+pic)
        self.img.setPixmap(QPixmap.fromImage(img))


class ChatWindowBubbleLeft(QWidget):
    def __init__(self, parent, **kwargs):
        # 默认宽度 640, 高度 480
        QtWidgets.QWidget.__init__(self)
        self.parent = parent

        # Window Config: Transparent
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 无边框 + 窗口置顶
        self.setAttribute(Qt.WA_TranslucentBackground)  # 半透明背景
        self.setAutoFillBackground(True)  # 非自动填充
        self.repaint()
        self.resize(256, 128)

        # Window Label
        self.bubble = QLabel(self)

        # auto-fade
        self.fade_out_timer = QTimer(self)
        self.fade_out_timer.timeout.connect(self.fade)
        self.fade_out_timer.setSingleShot(True)

        #
        self.animation = QPropertyAnimation(self.bubble, b"geometry")  # target, param, param 必须加 b

    def showMessage(self, message):
        self.animation.stop()
        self.bubble.resize(256, 128)
        self.bubble.setScaledContents(True)
        self.bubble.setAlignment(Qt.AlignCenter)
        self.bubble.setStyleSheet("border-image:url(img/chat_bb_l_640_548.png);")
        self.bubble.setText(message * 10)
        self.bubble.setWordWrap(True)
        self.move(self.parent.x() - self.bubble.width(), self.parent.y() - self.bubble.height())
        self.show()
        # 1s 后自动消失
        self.fade_out_timer.start(1500)

    def mousePressEvent(self, event):
        self.close()

    def fade(self):
        self.animation.setDuration(3000)
        self.animation.setEndValue(QRect(0, 0, 0, 0))
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.valueChanged.connect(self.on_value_changed)
        self.animation.finished.connect(self.on_finished)
        self.animation.start()

    def on_value_changed(self):
        # 根据动画进程的百分比，设置label的透明度
        opacity = 1.0 - self.animation.currentLoopTime() / self.animation.totalDuration()
        self.bubble.setWindowOpacity(opacity)

    def on_finished(self):
        self.close()


class ChatWindowNormal(QWidget):
    def __init__(self, parent, **kwargs):
        QtWidgets.QWidget.__init__(self)
        self.agent = parent.agent
        self.user_name = parent.user_name
        self.agent_name = parent.agent_name
        # 添加记忆存储
        self.setGeometry(parent.x() - 600, parent.y() + parent.height() - 337, 600, 337)
        self.setWindowTitle("Vega")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)  # 无边框 + 窗口置顶
        palette = QtGui.QPalette()
        bg = QtGui.QPixmap("img/background.jpg")
        palette.setBrush(self.backgroundRole(), QtGui.QBrush(bg))
        self.setPalette(palette)
        self.init_ui()
        self.work_thread()

    def init_ui(self):
        # 多行文本显示，显示所有的聊天信息
        self.content = QTextBrowser(self)
        self.content.setGeometry(30, 30, 550, 150)

        # 单行文本，消息发送框
        self.message = QLineEdit(self)
        self.message.setGeometry(30, 220, 550, 30)
        self.message.setPlaceholderText("请输入发送内容")

        # 发送按钮
        self.button = QPushButton("发送", self)
        self.button.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.button.setGeometry(520, 270, 60, 30)

    # 发送消息 + 接收消息
    def send_msg(self):
        msg = self.message.text()
        self.content.append(f"{self.user_name}: {msg}")

        if msg.upper() == "Q" or "退下吧" in msg:
            self.content.append(f"{self.agent_name}: 切~ 臭屁! 拜拜 👋")
            self.delay_to_do(self.do_close)
        else:
            continue_chat, text_output = self.agent.generate_dialogue_response(f"{self.user_name} 对 {self.agent_name} 说: {msg}")
            self.content.append(f"{self.agent_name}: {text_output}")
            if not continue_chat:
                self.delay_to_do(self.do_close)
        self.message.clear()

    def delay_to_do(self, slot):
        self.timer = QTimer(self)
        self.timer.timeout.connect(slot)
        self.timer.start(1000)

    def do_close(self):
        self.timer.stop()
        self.close()

    # 接收消息
    def recv_msg(self):
        while True:
            data = self.message.text().encode()
            print(type(data))
            if data != "" or data is not None:
                data = str(data) + "\n"
                self.content.append(data)
            else:
                exit()

    # 点击按钮发送消息
    def btn_send(self):
        self.button.clicked.connect(self.send_msg)

    # 线程处理
    def work_thread(self):
        Thread(target=self.btn_send).start()
        # Thread(target=self.recv_msg).start()

    def closeEvent(self, event):
        # self.destroy()
        self.close()
