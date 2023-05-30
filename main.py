import os
import sys
import json
import openai

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


"""
Vega Bootstrap

Vega: My first virtual companion


@date: 2023/4/26
@author: Yancy
"""

icon = os.path.join('img/icon.png')


class Vega(QWidget):
    def __init__(self):
        # 初始化基本属性
        QWidget.__init__(self)

        # Load config TODO: load user config from server
        config_private = 'config_private.json'
        self.config = {}
        with open(config_private, encoding='utf-8') as f:
            self.config = json.load(f)
        # set openai env
        api_key = self.config.get("openai_api_key")
        api_base = self.config.get("openai_api_base")
        openai.api_key = api_key
        openai.api_base = api_base
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = api_base

        self.agents = self.config.get("agents")
        self.user_id = self.config.get("user_id")
        self.user_name = self.config.get("user_name")

        # Tray Config
        # showing = QAction("现身~", self, triggered=self.showing)
        # showing.setIcon(QIcon(icon))
        quit = QAction("退出", self, triggered=self.quit)
        quit.setIcon(QIcon(icon))
        self.tray_icon_menu = QMenu(self)
        # self.tray_icon_menu.addAction(showing)
        self.tray_icon_menu.addAction(quit)
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(icon))
        self.tray_icon.setContextMenu(self.tray_icon_menu)
        self.tray_icon.show()

    def quit(self):
        # 保存记忆
        # self.agent.context.save_context_to_local(self.user_context_file)
        self.close()
        sys.exit()

    # 通过窗口透明度 显示/隐藏
    def showing(self):
        self.setWindowOpacity(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    vega = Vega()
    sys.exit(app.exec_())
