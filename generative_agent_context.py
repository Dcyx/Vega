import logging
import datetime
from typing import List

"""
聊天快照
@date: 2023/5/6
@author: Xiaohui
"""


logger = logging.getLogger(__name__)


class GenerativeAgentContext(object):
    """ 先把消息记录存内存，后面再选数据库进行优化 """
    context_stream: List[str] = []
    """ 聊天的上下文 """

    def add_context(self, observation: str):
        '''
        添加上下消息记录
        Args
            observation: 当前的观察
        '''
        current_time_str = datetime.datetime.now().strftime("%B %d, %Y, %I:%M %p")
        self.context_stream.append(f"{current_time_str}\t{observation}")

    def get_last_context(self, last_k=5):
        """
        获取最新 k 条消息记录
        Args
            last_k: 最后 k 条消息记录
        """
        last_context = []
        context = self.context_stream[-last_k:]

        for _context in context:
            context_time = _context.split("\t")[0]
            time_obj = datetime.datetime.strptime(context_time, '%B %d, %Y, %I:%M %p')
            current_time = datetime.datetime.now()
            time_diff = current_time - time_obj
            if time_diff < datetime.timedelta(hours=4):
                last_context.append(_context)
        if len(last_context) == 0:
            return "空"
        return "\n- ".join(last_context)

    def load_context_from_local(self, file_path):
        """
        load dialog context from local file
        """
        with open(file_path) as fr:
            for line in fr.readlines():
                self.context_stream.append(line.strip())

    def save_context_to_local(self, file_path):
        """
        save dialog context to local file
        """
        with open(file_path, 'w') as fw:
            for context in self.context_stream:
                fw.write(context)
                fw.write('\n')