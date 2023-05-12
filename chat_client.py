import sys
import time
import openai
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import *
from threading import Thread


class ChatClient(QWidget):
    def __init__(self, parent, **kwargs):
        QtWidgets.QWidget.__init__(self)
        self.agent = parent.agent
        self.user_name = parent.user_name
        self.agent_name = parent.agent_name
        # 添加记忆存储
        self.user_memory_dir = parent.user_memory_dir
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

    def get_completion(self, prompt):
        """

        """
        messages = [
            {"role": "system", "content": "接下来你要以 vega 的身份跟我说话，你的名字叫 vega，灵感来源于天琴座中最明亮的星星 织女一，"
                                          "同时织女也是中国古代爱情故事牛郎与织女中的角色。你的性格是非常活泼，幽默的。你作为我的朋友，"
                                          "会尽量满足我的要求，同时，我们是非常亲近的朋友，你有什么话都可以直说。不要用敬语，语气口吻越随便越好。"
                                          "接下来你只需要以我的朋友 vega 这个身份与我交流。"},
            {"role": "user", "content": prompt}
        ]
        try:
            completion = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages
            )
        except Exception as e:
            print(e)
            return "出错啦~"
        if 'choices' not in completion:
            print(completion)
            return "出错啦~"
        return completion.choices[0].message.content

    # 发送消息 + 接收消息
    def send_msg(self):
        msg = self.message.text()
        self.content.append(f"{self.user_name}: {msg}")

        if msg.upper() == "Q" or "退下吧" in msg:
            self.content.append(f"{self.agent_name}: 切~ 臭屁! 拜拜 👋")
            self.delay_to_do(self.do_destroy)
        else:
            continue_chat, text_output = self.agent.generate_dialogue_response(f"{self.user_name} 对 {self.agent_name} 说: {msg}")
            self.agent.memory.memory_retriever.save_memories_to_local(self.user_memory_dir)

            self.content.append(f"{self.agent_name}: {text_output}")
            if not continue_chat:
                self.delay_to_do(self.do_destroy)
        self.message.clear()

    def delay_to_do(self, slot):
        self.timer = QTimer(self)
        self.timer.timeout.connect(slot)
        self.timer.start(1000)

    def do_destroy(self):
        self.timer.stop()
        self.destroy()

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
        self.destroy()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = ChatClient()
    client.show()
    sys.exit(app.exec_())
