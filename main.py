import os
import sys
import math
import faiss
import codecs
import configparser
import openai
import random

from generative_agent import GenerativeAgent
from generative_agent_memory import GenerativeAgentMemory
from generative_agent_context import GenerativeAgentContext

from typing import Optional


from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets

from chat_client import ChatClient

from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
# from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS, milvus

from vector.vector_store_retriever import TimeWeightedVectorStoreRetriever



"""
Vega Bootstrap

Vega: My first virtual companion


@date: 2023/4/26
@author: Yancy
"""

icon = os.path.join('img/icon.png')
FAISS_DIR = "data/index"


def get_images(pics):
    pic_list = []
    for item in pics:
        img = QImage()
        img.load('img/'+item)
        pic_list.append(img)
    return pic_list


class Vega(QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.action = None
        self.index = None
        self.left_click = False
        self.mouse_drag_pos = None

        # Tray Config
        showing = QAction("现身~", self, triggered=self.showing)
        showing.setIcon(QIcon(icon))
        quit = QAction("退出", self, triggered=self.quit)
        quit.setIcon(QIcon(icon))
        self.tray_icon_menu = QMenu(self)
        self.tray_icon_menu.addAction(showing)
        self.tray_icon_menu.addAction(quit)
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(icon))
        self.tray_icon.setContextMenu(self.tray_icon_menu)
        self.tray_icon.show()

        # Vega Action Window Config
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 无边框 + 窗口置顶
        self.setAttribute(Qt.WA_TranslucentBackground)  # 半透明背景
        self.setAutoFillBackground(True)  # 非自动填充
        self.repaint()

        # Load actions & resize & init position
        self.img = QLabel(self)
        self.action_dataset = []
        self.init_data()
        self.set_pic("vega1.png")
        self.resize(128, 128)
        self.show()
        self.runing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_random_actions)
        self.timer.start(500)
        self.init_position(random_pos=False)

        # Load openai config
        config_private = 'config_private.ini'
        self.config = configparser.ConfigParser()
        with codecs.open(config_private, 'r', 'utf-8') as f:
            # 读取配置文件内容
            self.config = configparser.ConfigParser()
            self.config.read_file(f)
        openai.api_key = self.config.get("OpenAI", "api_key")
        openai.api_base = self.config.get("OpenAI", "api_base")
        os.environ["OPENAI_API_KEY"] = self.config.get("OpenAI", "api_key")
        os.environ["OPENAI_API_BASE"] = self.config.get("OpenAI", "api_base")

        # TODO 用户id应该是生成的不应该是写死的，应该支持多用户
        self.user_name = self.config.get("User", "name")
        self.user_id = self.config.get("User", "id")

        self.agent_name = self.config.get("Agent", "name")
        self.age = self.config.get("Agent", "age")
        traits = self.config.get("Agent", "traits")
        relation = self.config.get("Agent", "relation")

        # Init generative vector
        language_model = ChatOpenAI(max_tokens=1500, model_name="gpt-3.5-turbo")  # Can be any LLM you want.

        # 先根据 id 判断当前用户的记忆是否存在
        self.user_memory_dir = os.path.join(FAISS_DIR, self.user_id)
        if not os.path.exists(self.user_memory_dir):
            os.makedirs(self.user_memory_dir, exist_ok=True)

        if len(os.listdir(self.user_memory_dir)) == 0:
            vega_memory = GenerativeAgentMemory(
                llm=language_model,
                memory_retriever=self.create_new_memory_retriever(load_memory=False),
                verbose=True,
                reflection_threshold=8  # we will give this a relatively low number to show how reflection works
            )
        else:
            print(f"load memory from {self.user_memory_dir}")
            vega_memory = GenerativeAgentMemory(
                llm=language_model,
                memory_retriever=self.create_new_memory_retriever(load_memory=True, user_memory_dir=self.user_memory_dir),
                verbose=True,
                reflection_threshold=8
            )
        vega_context = GenerativeAgentContext()
        self.agent = GenerativeAgent(
            name=self.agent_name,
            age=self.age,
            traits=traits,
            relation=relation,
            llm=language_model,
            memory=vega_memory,
            context=vega_context,
            verbose=True
        )

    def init_data(self):
        # singing
        imgs = get_images(["vega1.png", "vega1b.png", "vega2.png", "vega2b.png", "vega3.png", "vega3b.png"])
        self.action_dataset.append(imgs)
        #
        imgs = get_images(["vega11.png", "vega15.png", "vega16.png", "vega17.png", "vega16.png", "vega17.png", "vega16.png", "vega17.png"])
        self.action_dataset.append(imgs)
        imgs = get_images(["vega54.png", "vega55.png", "vega26.png", "vega27.png", "vega28.png", "vega29.png", "vega26.png", "vega27.png", "vega28.png", "vega29.png", "vega26.png", "vega27.png", "vega28.png", "vega29.png"])
        self.action_dataset.append(imgs)
        imgs = get_images(["vega31.png", "vega32.png", "vega31.png", "vega33.png"])
        self.action_dataset.append(imgs)
        imgs = get_images(["vega18.png", "vega19.png"])
        self.action_dataset.append(imgs)
        imgs = get_images(["vega34b.png", "vega35b.png", "vega34b.png", "vega36b.png"])
        self.action_dataset.append(imgs)
        imgs = get_images(["vega14.png", "vega14.png", "vega52.png", "vega13.png", "vega13.png", "vega13.png", "vega52.png", "vega14.png"])
        self.action_dataset.append(imgs)
        imgs = get_images(["vega42.png", "vega43.png", "vega44.png", "vega45.png", "vega46.png"])
        self.action_dataset.append(imgs)
        imgs = get_images(["vega1.png", "vega38.png", "vega39.png", "vega40.png", "vega41.png"])
        self.action_dataset.append(imgs)
        imgs = get_images(["vega25.png", "vega25.png", "vega53.png", "vega24.png", "vega24.png", "vega24.png", "vega53.png", "vega25.png"])
        self.action_dataset.append(imgs)
        imgs = get_images(["vega20.png", "vega21.png", "vega20.png", "vega21.png", "vega20.png"])
        self.action_dataset.append(imgs)

    def set_pic(self, pic):
        img = QImage()
        img.load('img/'+pic)
        self.img.setPixmap(QPixmap.fromImage(img))

    def run_random_actions(self):
        if not self.runing:
            self.action = random.randint(0, len(self.action_dataset)-1)
            self.index = 0
            self.runing = True
        imgs = self.action_dataset[self.action]
        if self.index >= len(imgs):
            self.index = 0
            self.runing = False
        self.img.setPixmap(QPixmap.fromImage(imgs[self.index]))
        self.index += 1

    def init_position(self, random_pos=False):
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - self.geometry().width()) * (random.random() if random_pos else 1)
        y = (screen.height() - self.geometry().height()) * (random.random() if random_pos else 1)
        self.move(x, y)

    def quit(self):
        # 保存记忆
        self.close()
        sys.exit()

    # 通过窗口透明度 显示/隐藏
    def showing(self):
        self.setWindowOpacity(1)

    # 鼠标左键按下时, 宠物将和鼠标位置绑定
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_click = True
        self.mouse_drag_pos = event.globalPos() - self.pos()
        event.accept()
        # 拖动时鼠标图形的设置
        self.setCursor(QCursor(Qt.OpenHandCursor))

    # 鼠标移动时调用，实现宠物随鼠标移动
    def mouseMoveEvent(self, event):
        # 如果鼠标左键按下，且处于绑定状态
        if Qt.LeftButton and self.left_click:
            # 宠物随鼠标进行移动
            self.move(event.globalPos() - self.mouse_drag_pos)
        event.accept()

    # 鼠标释放调用，取消绑定
    def mouseReleaseEvent(self, event):
        self.left_click = False
        # 鼠标图形设置为箭头
        self.setCursor(QCursor(Qt.ArrowCursor))

    def enterEvent(self, event):
        self.setCursor(Qt.ClosedHandCursor)  # 设置鼠标形状 Qt.ClosedHandCursor 非指向手

    # 右键点击交互
    def contextMenuEvent(self, event):
        # 定义菜单
        menu = QMenu(self)
        # 定义菜单项
        question_answer = menu.addAction("聊聊?")
        hide = menu.addAction("退下吧~")
        menu.addSeparator()

        # 使用exec_()方法显示菜单。从鼠标右键事件对象中获得当前坐标。mapToGlobal()方法把当前组件的相对坐标转换为窗口（window）的绝对坐标。
        action = menu.exec_(self.mapToGlobal(event.pos()))
        # if action == quitAction:
        #     qApp.quit()
        if action == hide:
            self.setWindowOpacity(0)
        if action == question_answer:
            self.client = ChatClient(parent=self)
            self.client.show()

    #
    def on_widget2_position(self):
        widget2_pos = vega.pos()
        print(widget2_pos)


    def relevance_score_fn(self, score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # This will differ depending on a few things:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
        # This function converts the euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - score / math.sqrt(2)

    def create_new_memory_retriever(self, load_memory=False, user_memory_dir=None):
        """Create a new vector store retriever unique to the vector."""
        # Define your embedding model
        embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index,
                            InMemoryDocstore({}), {}, relevance_score_fn=self.relevance_score_fn)
        retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)

        if load_memory:
            retriever.load_memories_from_local(user_memory_dir, embeddings_model)
        return retriever


if __name__ == '__main__':
    app = QApplication(sys.argv)
    vega = Vega()
    sys.exit(app.exec_())

