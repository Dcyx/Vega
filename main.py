import os
import sys
import json
import openai
import random
from functools import partial
from agent import Agent

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

        # Load Config
        config_private = 'config_private.json'
        self.config = {}
        with open(config_private, encoding='utf-8') as f:
            self.config = json.load(f)

        # Set openai env
        api_key = self.config.get("openai_api_key")
        api_base = self.config.get("openai_api_base")
        openai.api_key = api_key
        openai.api_base = api_base
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = api_base

        # Init user info
        self.user_id = self.config.get("user_id")
        self.user_name = self.config.get("user_name")
        self.agent_config_list = self.config.get("agents")
        self.agents = []

        # Tray Config
        action_quit = QAction("quit", self, triggered=self.quit)
        action_quit.setIcon(QIcon(icon))

        self.tray_icon_menu = QMenu(self)
        menu_agents = self.tray_icon_menu.addMenu("agents")
        menu_agents.setIcon(QIcon(icon))
        for agent in self.agent_config_list:
            _action_show_agent = QAction(agent["name"], self)
            _action_show_agent.setData(agent)
            _action_show_agent.triggered.connect(self.showing)
            _action_show_agent.setIcon(QIcon(icon))
            menu_agents.addAction(_action_show_agent)
        self.tray_icon_menu.addAction(action_quit)
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(icon))
        self.tray_icon.setContextMenu(self.tray_icon_menu)
        self.tray_icon.show()


    def quit(self):
        # 保存记忆
        # self.agent.context.save_context_to_local(self.user_context_file)
        self.close()
        sys.exit()

    #
    def showing(self):
        agent_config = self.sender().data()
        agent_id = agent_config["id"]
        cached = False
        for agent in self.agents:
            if agent.agent_id == agent_id:
                agent.setWindowOpacity(1)
                agent.show()
                cached = True
                break
        if not cached:
            agent = Agent(self.user_id, self.user_name, agent_config)
            agent.setWindowOpacity(1)
            agent.show()
            self.agents.append(agent)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    vega = Vega()
    sys.exit(app.exec_())
