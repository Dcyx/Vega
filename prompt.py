import time
from typing import Any, Callable, List

from pydantic import BaseModel
from prompt_generator import get_prompt

from langchain.prompts.chat import (
    BaseChatPromptTemplate,
)
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever
"""
shoot for moon


@date: 2023/5/11
@author: Yancy
"""


class IntelligentAgentPrompt(BaseChatPromptTemplate, BaseModel):
    ai_name: str
    ai_role: str
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196

    def construct_full_prompt(self, goals: List[str]) -> str:
        # Construct full prompt
        full_prompt = (
            f"You are {self.ai_name}, {self.ai_role}\n\nGOALS:\n\n"
        )
        for i, goal in enumerate(goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """
        参数构造顺序: 关于我的描述, 相关记忆, 对话上下文, 目标, user_content
        :param kwargs: 参数 + 目标, 不同目标会有不同的参数, 构造逻辑
        :return:
        """
        base_prompt = SystemMessage(content=self.construct_full_prompt(kwargs["goals"]))
        time_prompt = SystemMessage(content=f"当前时间是 {time.strftime('%c')}")
        used_tokens = self.token_counter(base_prompt.content) + self.token_counter(
            time_prompt.content
        )
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]  # 对话上下文
        relevant_docs = memory.get_relevant_documents(str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [self.token_counter(doc) for doc in relevant_memory]
        )
        while used_tokens + relevant_memory_tokens > 2500:
            relevant_memory = relevant_memory[:-1]
            relevant_memory_tokens = sum(
                [self.token_counter(doc) for doc in relevant_memory]
            )
        content_format = (
            f"This reminds you of these events "
            f"from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens
        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[BaseMessage] = [base_prompt, time_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages
