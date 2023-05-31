import os
import sys
import time
import math
import json
import random
import openai
import traceback

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QMenu
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QRunnable
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QThreadPool
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QCursor

from chat_client import WindowBubble
from chat_client import ThoughtWindowBubble
from chat_client import ChatWindowBubbleLeft
from chat_client import ChatWindowBubbleRight
from chat_client import ChatWindowNormal

import speech_recognition as sr

from generative_agent import GenerativeAgent
from generative_agent_memory import GenerativeAgentMemory
from generative_agent_context import GenerativeAgentContext

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from vectorstores import Milvus
from vector.vector_store_retriever import TimeWeightedVectorStoreRetriever

"""
Agent UI

@date: 2023/5/30
@author: Yancy
"""

# TODO: 第二轮识别中异常退出

CONTEXT_DIR = "data/context"


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the vector."""

    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()

    # Initialize the vectorstore as empty
    vectorstore = Milvus(embeddings_model, connection_args={"host": "127.0.0.1", "port": "19530"}, relevance_score_fn=relevance_score_fn)
    retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)
    return retriever


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def listen_microphone_thread():
    print("enter listen:" + time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())))
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000
    work_microphones = sr.Microphone.list_working_microphones()
    key_list = list(work_microphones.keys())
    # Yancy TODO: 校验是否有 work microphone, 要抛异常
    with sr.Microphone(key_list[0]) as source:
        print("----正在听")
        print("begin listen:" + time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())))
        audio = recognizer.listen(source, timeout=15)  # Yancy: 这里要加超时, 否则由于背景噪音过大, 会一直监听
        print("----听完啦")
        return audio


def recognize_audio_thread(audio):
    try:
        print("识别中")
        recognizer = sr.Recognizer()
        recognized_text = recognizer.recognize_whisper(audio, model="medium", language="zh")
        print("识别完成")
        recognized_text = recognized_text.strip()
        if len(recognized_text) > 0:
            return recognized_text
        else:
            return "未检测到语音"
    except sr.UnknownValueError:
        print("Could not understand audio")
        return "无法理解"
    except sr.RequestError as e:
        print("Error:", str(e))
        return "网络异常"
    except Exception as e:
        print(str(e))
        return "其他异常"


def show_bubble_message(bubble_type, px, py, message):
    bubble: WindowBubble
    if bubble_type == "thought":
        bubble = ThoughtWindowBubble(px, py)
    elif bubble_type == "chat_left":
        bubble = ChatWindowBubbleLeft(px, py)
    elif bubble_type == "chat_right":
        bubble = ChatWindowBubbleRight(px, py)
    else:
        bubble = ThoughtWindowBubble(px, py)
    bubble.show_message(message)


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    status
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    status = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.status.emit()  # Done


class Agent(QWidget):
    def __init__(self, user_id, user_name, agent_config):
        QWidget.__init__(self)

        # Agent ui params
        self.left_click = False
        self.mouse_drag_pos = None

        # Agent character params
        self.user_name = user_name
        self.user_id = user_id
        self.agent_id = agent_config["id"]
        self.agent_name = agent_config['name']
        self.agent_age = agent_config['age']
        self.agent_traits = agent_config['traits']
        self.agent_relation = agent_config['relation']

        # 初始化线程池相关对象
        self.worker_listen_result = None
        self.worker_listen_finished = True
        self.worker_recognize_result = None
        self.worker_recognize_finished = True
        self.worker_response_result = None
        self.worker_response_finished = True
        self.thread_pool = QThreadPool()

        # UI Window Config
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 无边框 + 窗口置顶
        self.setAttribute(Qt.WA_TranslucentBackground)  # 半透明背景
        self.setAutoFillBackground(True)  # 非自动填充
        self.repaint()

        # Load actions & resize & init position
        self.action_runing = False
        self.action_index = None
        self.action_img_index = None

        self.img = QLabel(self)
        self.action_dataset = []
        self.load_agent_imgs()
        self.set_pic("vega1.png")
        self.resize(128, 128)
        self.random_action_timer = QTimer()
        self.random_action_timer.timeout.connect(self.run_random_actions)
        self.random_action_timer.start(500)
        self.init_position(random_pos=False)

        # Init generative vector
        language_model = ChatOpenAI(max_tokens=1500, model_name="gpt-3.5-turbo")  # Can be any LLM you want.

        #
        vega_memory = GenerativeAgentMemory(
            llm=language_model,
            memory_retriever=create_new_memory_retriever(),
            verbose=True,
            reflection_threshold=8  # we will give this a relatively low number to show how reflection works
        )

        #
        user_context_dir = os.path.join(CONTEXT_DIR, user_id, self.agent_name)
        user_context_file = os.path.join(user_context_dir, "context.txt")
        if not os.path.exists(user_context_dir):
            os.makedirs(user_context_dir, exist_ok=True)
        vega_context = GenerativeAgentContext()
        if os.path.exists(user_context_file):
            vega_context.load_context_from_local(user_context_file)

        self.agent = GenerativeAgent(
            name=self.agent_name,
            age=self.agent_age,
            traits=self.agent_traits,
            relation=self.agent_relation,
            llm=language_model,
            memory=vega_memory,
            context=vega_context,
            verbose=True
        )

        # Long press event, start a speech recognition listener
        self.long_press_timer = QTimer(self)
        self.long_press_timer.timeout.connect(self.on_long_press)
        self.long_press_timer.setSingleShot(True)

        #
        self.chat_window_norm = ChatWindowNormal(parent=self)

    def worker_listen_result_post_process(self, result):
        self.worker_listen_result = result

    def worker_listen_status_post_process(self):
        self.worker_listen_finished = True

    def worker_recognize_result_post_process(self, result):
        self.worker_recognize_result = result

    def worker_recognize_status_post_process(self):
        self.worker_recognize_finished = True

    def worker_response_result_post_process(self, result):
        self.worker_response_result = result

    def worker_response_status_post_process(self):
        self.worker_response_finished = True

    def response_speech_thread(self, observation):
        continue_chat, text_output = self.agent.generate_dialogue_response(observation)
        return text_output
        
    def get_images(self, pics):
        pic_list = []
        for item in pics:
            img = QImage()
            img.load('img/' + item)
            pic_list.append(img)
        return pic_list
    
    def load_agent_imgs(self):
        # singing
        imgs = self.get_images(["vega1.png", "vega1b.png", "vega2.png", "vega2b.png", "vega3.png", "vega3b.png"])
        self.action_dataset.append(imgs)
        #
        imgs = self.get_images(["vega11.png", "vega15.png", "vega16.png", "vega17.png", "vega16.png", "vega17.png", "vega16.png", "vega17.png"])
        self.action_dataset.append(imgs)
        imgs = self.get_images(["vega54.png", "vega55.png", "vega26.png", "vega27.png", "vega28.png", "vega29.png", "vega26.png", "vega27.png", "vega28.png", "vega29.png", "vega26.png", "vega27.png", "vega28.png", "vega29.png"])
        self.action_dataset.append(imgs)
        imgs = self.get_images(["vega31.png", "vega32.png", "vega31.png", "vega33.png"])
        self.action_dataset.append(imgs)
        imgs = self.get_images(["vega18.png", "vega19.png"])
        self.action_dataset.append(imgs)
        imgs = self.get_images(["vega34b.png", "vega35b.png", "vega34b.png", "vega36b.png"])
        self.action_dataset.append(imgs)
        imgs = self.get_images(["vega14.png", "vega14.png", "vega52.png", "vega13.png", "vega13.png", "vega13.png", "vega52.png", "vega14.png"])
        self.action_dataset.append(imgs)
        imgs = self.get_images(["vega42.png", "vega43.png", "vega44.png", "vega45.png", "vega46.png"])
        self.action_dataset.append(imgs)
        imgs = self.get_images(["vega1.png", "vega38.png", "vega39.png", "vega40.png", "vega41.png"])
        self.action_dataset.append(imgs)
        imgs = self.get_images(["vega25.png", "vega25.png", "vega53.png", "vega24.png", "vega24.png", "vega24.png", "vega53.png", "vega25.png"])
        self.action_dataset.append(imgs)
        imgs = self.get_images(["vega20.png", "vega21.png", "vega20.png", "vega21.png", "vega20.png"])
        self.action_dataset.append(imgs)

    def set_pic(self, pic):
        img = QImage()
        img.load('img/'+pic)
        self.img.setPixmap(QPixmap.fromImage(img))

    def run_random_actions(self):
        if not self.action_runing:
            self.action_index = random.randint(0, len(self.action_dataset) - 1)
            self.action_img_index = 0
            self.action_runing = True
        imgs = self.action_dataset[self.action_index]
        if self.action_img_index >= len(imgs):
            self.action_img_index = 0
            self.action_runing = False
        self.img.setPixmap(QPixmap.fromImage(imgs[self.action_img_index]))
        self.action_img_index += 1

    def init_position(self, random_pos=False):
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - self.geometry().width()) * (random.random() if random_pos else 1)
        y = (screen.height() - self.geometry().height()) * (random.random() if random_pos else 1)
        self.move(x, y)

    def on_long_press(self):
        """长按触发语音聊天
        ʕ̡̢̡ʘ̅͟͜͡ʘ̲̅ʔ̢̡̢ ʕ•̫͡•ʕ*̫͡*ʕ•͓͡•ʔ-̫͡-ʕ•̫͡•ʔ*̫͡*ʔ-̫͡-ʔ

        `GUI 优先级较低, 主线程占用会导致 GUI 卡死
        `不可避免的阻塞逻辑中 可以通过 QApplication.processEvents() 主动调用处理事件
        `https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/
        """
        print("进入长按事件")
        # 启动监听
        listen_worker = Worker(listen_microphone_thread)
        listen_worker.signals.result.connect(self.worker_listen_result_post_process)
        listen_worker.signals.status.connect(self.worker_listen_status_post_process)
        self.worker_listen_finished = False
        self.thread_pool.start(listen_worker)

        # 提示用户: 开始对话
        print("打开 Bubble: 监听")
        show_bubble_message("thought", self.x(), self.y(), "<p>" + "让我听听是谁在说话 (¯。¯ԅ)" + "</p>")

        # 等待监听结果
        # time.sleep(0.5)
        while True:
            if self.worker_listen_finished:
                break
            else:
                QApplication.processEvents()
                time.sleep(0.016)
        print("拿到语音数据")
        audio = self.worker_listen_result

        # 启动识别
        recognize_worker = Worker(recognize_audio_thread, audio)
        recognize_worker.signals.result.connect(self.worker_recognize_result_post_process)
        recognize_worker.signals.status.connect(self.worker_recognize_status_post_process)
        self.worker_recognize_finished = False
        self.thread_pool.start(recognize_worker)

        while True:
            if self.worker_recognize_finished:
                break
            else:
                QApplication.processEvents()
                time.sleep(0.016)
        print("拿到语音结果")
        recognized_text = self.worker_recognize_result

        # 提示用户: 语音识别结果
        show_bubble_message("thought", self.x(), self.y(), "<p>" + f"{self.user_name} 说: {recognized_text}" + "</p>")

        # 调用语言模型
        response_worker = Worker(self.response_speech_thread, recognized_text)
        response_worker.signals.result.connect(self.worker_response_result_post_process)
        response_worker.signals.status.connect(self.worker_response_status_post_process)
        self.worker_response_finished = False
        self.thread_pool.start(response_worker)

        # 等待语言模型返回结果
        while True:
            if self.worker_response_finished:
                break
            else:
                QApplication.processEvents()
                time.sleep(0.016)
        print("拿到语言模型返回结果")
        response_text = self.worker_response_result

        show_bubble_message("chat_right", self.x(), self.y(), "<p>" + response_text + "</p>")

    # 鼠标按下事件 不分左右, 将和鼠标位置绑定
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_click = True
            self.mouse_drag_pos = event.globalPos() - self.pos()  # 鼠标点击位置 - agent 位置
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            self.long_press_timer.start(300)
        event.accept()

    # 鼠标双击事件
    def mouseDoubleClickEvent(self, event):
        # 停止长按计时
        self.long_press_timer.stop()
        # 打开聊天窗口
        self.chat_window_norm.setGeometry(self.x() - 600, self.y() + self.height() - 337, 600, 337)
        self.chat_window_norm.show()

    # 鼠标按下后的移动事件
    def mouseMoveEvent(self, event):
        # 如果鼠标左键按下，且处于绑定状态
        if self.left_click and (event.globalPos() - self.mouse_drag_pos - self.pos()).x() != 0:
            #
            self.setCursor(Qt.ClosedHandCursor)
            # 随鼠标进行移动
            self.move(event.globalPos() - self.mouse_drag_pos)
            # 明显出现位移后再停止长按计时
            self.long_press_timer.stop()
            event.accept()

    # 鼠标释放调用
    def mouseReleaseEvent(self, event):
        # 恢复初始状态
        self.left_click = False
        self.setCursor(QCursor(Qt.ArrowCursor))
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
            self.chat_window_norm.setGeometry(self.x() - 600, self.y() + self.height() - 337, 600, 337)
            self.chat_window_norm.show()

    # 选中的前提下, 鼠标进入事件
    def enterEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)  # 设置鼠标形状 Qt.ClosedHandCursor 非指向手


if __name__ == '__main__':
    app = QApplication(sys.argv)
    config_private = 'config_private.json'
    config = {}
    with open(config_private, encoding='utf-8') as f:
        config = json.load(f)
    # set openai env
    api_key = config.get("openai_api_key")
    api_base = config.get("openai_api_base")
    openai.api_key = api_key
    openai.api_base = api_base
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    vega = Agent("888", "Yancy", {
      "name": "Vega",
      "age": 24,
      "traits": "名字灵感来源于天琴座中最亮的卫星-织女一。性格活泼、幽默，平时大大咧咧，神经很大条，但同时又有很强的同理心，偶尔也会很温柔。喜欢开玩笑、玩网络梗、偶尔也会调侃朋友。",
      "relation": "Vega 与 Yancy 是朋友关系"
    })
    vega.show()
    sys.exit(app.exec_())
    