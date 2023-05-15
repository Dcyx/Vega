'''
聊天快照
@date: 2023/5/6
@author: Xiaohui
'''

import logging
from datetime import datetime
import re
from typing import Any, Dict, List, Optional
from pydantic import Field


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
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        self.context_stream.append(f"{current_time_str}\t{observation}")

    def get_last_context(self, last_k=5):
        '''
        获取最新 k 条消息记录
        Args
            last_k: 最后 k 条消息记录
        '''
        context = self.context_stream[-last_k:]
        if len(context) == 0:
            return "空"
        return "\n- " + "\n- ".join(context)

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