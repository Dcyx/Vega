import os
import sys
import math
import codecs
import configparser
import openai
import random
import speech_recognition as sr

from generative_agent import GenerativeAgent
from generative_agent_memory import GenerativeAgentMemory
from generative_agent_context import GenerativeAgentContext

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets

from vectorstores import Milvus
from vector.vector_store_retriever import TimeWeightedVectorStoreRetriever

from chat_client import ChatClient

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


"""
Vega Bootstrap

Vega: My first virtual companion


@date: 2023/4/26
@author: Yancy
"""

icon = os.path.join('img/icon.png')
FAISS_DIR = "data/index"
CONTEXT_DIR = "data/context"


def get_images(pics):
    pic_list = []
    for item in pics:
        img = QImage()
        img.load('img/'+item)
        pic_list.append(img)
    return pic_list


def on_long_press():
    print("----Yancy----\nSpeechRecognition")
    #
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print(recognizer.energy_threshold)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Processing...")
        recognized_text = recognizer.recognize_whisper(audio, model="base")
        print("Recognized Text:", recognized_text)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error:", str(e))
    except Exception as e:
        print(str(e))


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the vector."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    '''
    change Faiss to Milvus which cause speech recognition error(torch.load(model))  
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index,
                        InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    '''
    vectorstore = Milvus(embeddings_model, connection_args={"host": "127.0.0.1", "port": "19530"}, relevance_score_fn=relevance_score_fn)
    retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)

    return retriever


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
        self.load_agent_imgs()
        self.set_pic("vega1.png")
        self.resize(128, 128)
        self.show()
        self.runing = False
        self.random_action_timer = QTimer()
        self.random_action_timer.timeout.connect(self.run_random_actions)
        self.random_action_timer.start(500)
        self.init_position(random_pos=False)

        # Long press event, start a speech recognition listener
        self.long_press_timer = QTimer(self)
        self.long_press_timer.timeout.connect(on_long_press)
        self.long_press_timer.setSingleShot(True)

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
        self.user_context_dir = os.path.join(CONTEXT_DIR, self.user_id)
        self.user_context_file = os.path.join(self.user_context_dir, "context.txt")
        if not os.path.exists(self.user_context_dir):
            os.makedirs(self.user_context_dir, exist_ok=True)

        vega_memory = GenerativeAgentMemory(
            llm=language_model,
            memory_retriever=create_new_memory_retriever(),
            verbose=True,
            reflection_threshold=8  # we will give this a relatively low number to show how reflection works
        )
        vega_context = GenerativeAgentContext()
        if os.path.exists(self.user_context_file):
            vega_context.load_context_from_local(self.user_context_file)

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

        # 最后初始化聊天面板: client 中引用了 self.agent
        self.client = ChatClient(parent=self)

    def load_agent_imgs(self):
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
        self.agent.context.save_context_to_local(self.user_context_file)
        self.close()
        sys.exit()

    # 通过窗口透明度 显示/隐藏
    def showing(self):
        self.setWindowOpacity(1)

    # 鼠标按下事件 不分左右, 将和鼠标位置绑定
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_click = True
            self.mouse_drag_pos = event.globalPos() - self.pos()  # 鼠标点击位置 - agent 位置
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            print("----Yancy----\nPress")
            self.long_press_timer.start(300)
        event.accept()

    # 鼠标双击事件
    def mouseDoubleClickEvent(self, event):
        print("----Yancy----\nDoubleClick")
        # 停止长按计时
        self.long_press_timer.stop()
        # 打开聊天窗口
        self.client.show()


    # 鼠标按下后的移动事件
    def mouseMoveEvent(self, event):
        # 如果鼠标左键按下，且处于绑定状态
        if self.left_click and (event.globalPos() - self.mouse_drag_pos - self.pos()).x() != 0:
            # 宠物随鼠标进行移动
            self.move(event.globalPos() - self.mouse_drag_pos)
            print("----Yancy----\nMove")
            # 明显出现位移后再停止长按计时
            self.long_press_timer.stop()
            event.accept()

    # 鼠标释放调用
    def mouseReleaseEvent(self, event):
        # 恢复初始状态
        self.left_click = False
        self.setCursor(QCursor(Qt.ArrowCursor))
        print("----Yancy----\nRelease")
        # 停止长按计时
        self.long_press_timer.stop()

    # 右键菜单事件
    def contextMenuEvent(self, event):
        # 停止长按计时
        self.long_press_timer.stop()
        # 配置右键菜单
        menu = QMenu(self)
        chat_with_me = menu.addAction("聊聊?")
        hide = menu.addAction("退下吧~")
        menu.addSeparator()
        # 使用exec_()方法显示菜单。从鼠标右键事件对象中获得当前坐标。mapToGlobal()方法把当前组件的相对坐标转换为窗口（window）的绝对坐标。
        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == hide:
            self.setWindowOpacity(0)
        if action == chat_with_me:
            self.client.show()

    # 选中的前提下, 鼠标进入事件
    def enterEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)  # 设置鼠标形状 Qt.ClosedHandCursor 非指向手
        print("----Yancy----\nEnter")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    vega = Vega()
    sys.exit(app.exec_())

