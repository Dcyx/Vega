import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from generative_agent_memory import GenerativeAgentMemory
from generative_agent_context import GenerativeAgentContext
from langchain.prompts import PromptTemplate

"""
Generative Agents reproduce based on LangChain 

https://arxiv.org/pdf/2304.03442.pdf

@date: 2023/5/6
@author: Yancy
"""


class GenerativeAgent(BaseModel):
    """A character with memory and innate characteristics."""

    name: str
    """The character's name."""

    age: Optional[int] = None
    """The optional age of the character."""
    traits: str = "N/A"
    """Permanent traits to ascribe to the character."""
    relation: str
    """The relationship to the speaker."""
    emotions: Optional[str] = None
    emotion_status: str = None
    """可选表情"""

    memory: GenerativeAgentMemory
    """The memory object that combines relevance, recency, and 'importance'."""
    context: GenerativeAgentContext

    llm: BaseLanguageModel
    """The underlying language model."""
    verbose: bool = False
    summary: str = ""  #: :meta private:
    """Stateful self-summary generated via reflection on the character's memory."""

    summary_refresh_seconds: int = 3600  #: :meta private:
    """How frequently to re-generate the summary."""

    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    """The last time the character's summary was regenerated."""

    daily_summaries: List[str] = Field(default_factory=list)  # : :meta private:
    """Summary of the events in the plan that the vector took."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    # LLM-related methods
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )

    def get_context(self):
        """获取聊天的上文"""
        return self.context.get_last_context(last_k=5)

    def _generate_reaction(self, observation: str) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "我的信息: {agent_description} "
            + "\n当前时间是: {current_time}."
            # + "\n当前场景下对 {agent_name} 的要求包括: {agent_claim}"
            + "\n相关记忆包括: {relevant_memories}"  # chain.run.prep_inputs 中检索 relevant_memories
            + "\n最近的聊天记录包括：{last_context}"
            # + "\n\n回复格式: 如果回复内容中包含告别的语义, 例如 拜拜、再见、下次再聊等, 就回复 'GOODBYE:\"回复内容\"'; 如果要继续对话, 就回复 'SAY:\"回复内容\"'. "
              "\n不要直接将背景信息作为回复,回答的简洁一点,不要啰嗦,注意对话上下文."
            + "\n\n基于观测到的聊天信息 '{observation}', {agent_name} 会回复什么内容? 会表现出什么表情？会有什么样的心理活动？心理活动以 {agent_name} 的视角输出。可选 Emotion 有 [" + self.emotions + "]"
            + "\n请输出一个 json，包含三个部分，回复内容、表情和动作，json 的 key 分别为 \"Response\" \"Emotion\" 和 \"Reaction\""
        )
        agent_description = self.get_agent_description()
        # agent_claim, obs_summary = self.get_agent_claim(agent_description=agent_description, observation=observation)
        last_context = self.get_context()
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs: Dict[str, Any] = dict(
            agent_description=agent_description,
            current_time=current_time_str,
            agent_name=self.name,
            # agent_claim=agent_claim,
            last_context=last_context,
            observation=observation,
        )
        # json 格式解析，如果解析失败则再试一次，最多试3次
        # TODO 尝试次数改配置
        try_resp = 0
        parser_response = None
        while try_resp < 3:
            response = self.chain(prompt=prompt).run(queries=[observation, ], **kwargs).strip()
            try: # 成功解析json 格式则跳出循环不再继续尝试
                parser_response = json.loads(response)
                break
            except:
                print(f'parser json failed, retry with try_resp = {try_resp}....')
                try_resp += 1
        print(f"parser_response = \n{parser_response}\n")

        return parser_response

    def get_agent_claim(self, agent_description, observation: str) -> str:
        """
        对当前情境进行抽象. 检索相关记忆并提炼 agent_claim, 返回 claim 和 observation 的抽象
        """
        # TODO:  backward,forward
        prompt_obs_summary = PromptTemplate.from_template(
            "{observation}"
            "\n基于以上对话内容,对消息中的场景、人物关系、事件等进行抽象, 总结成一句话. 不要过程,直接给结论."
            "\n\n"
        )
        obs_summary = self.chain(prompt_obs_summary).run(observation=observation).strip()
        obs_summary = re.sub(r"(\n)+", "; ", obs_summary).strip()
        print(f"----Yancy----\n{obs_summary}\n")

        # TODO: 优化 agent 目标定义
        prompt = PromptTemplate.from_template(
            "个人信息:{agent_description}"
            "\n相关记忆:"
            "\n{relevant_memories}"
            "\n基于自身信息和相关记忆, 推测 {agent_name} 面对情境 '{scene}' 中的内容, 要如何做才能既符合个人性格又能满足情境下可能的诉求?不要过程,直接给结论."
            "\n\n"
        )
        result = self.chain(prompt).run(queries=[obs_summary], agent_description=agent_description, scene=obs_summary, agent_name=self.name).strip()
        print(f"----Yancy----\n{result}\n")
        return result, obs_summary

    def _clean_response(self, text: str) -> str:
        return re.sub(f"^{self.name} ", "", text.strip()).strip()

    # def generate_reaction(self, observation: str) -> Tuple[bool, str]:
    #     """React to a given observation."""
    #     call_to_action_template = (
    #         "Should {agent_name} react to the observation, and if so,"
    #         + " what would be an appropriate reaction? Respond in one line."
    #         + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
    #         + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
    #         + "\nEither do nothing, react, or say something but not both.\n\n"
    #     )
    #     full_result = self._generate_reaction(observation, call_to_action_template)
    #     result = full_result.strip().split("\n")[0]
    #     # AAA
    #     self.memory.save_context(
    #         {},
    #         {
    #             self.memory.add_memory_key: f"{self.name} observed "
    #             f"{observation} and reacted by {result}"
    #         },
    #     )
    #     if "REACT:" in result:
    #         reaction = self._clean_response(result.split("REACT:")[-1])
    #         return False, f"{self.name} {reaction}"
    #     if "SAY:" in result:
    #         said_value = self._clean_response(result.split("SAY:")[-1])
    #         return True, f"{said_value}"
    #     else:
    #         return False, result

    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str, str]:
        """React to a given observation."""
        # 返回的是已经解析的 json 格式 or None
        response_raw = self._generate_reaction(observation)
        if response_raw is None:
            return True, "（网络出错了，请重试吧）"

        # response = response_raw.strip().split("\n")[0]
        try:
            response, emotion = response_raw['Response'], response_raw['Emotion']
            print(f'> response = {response}, emotion = {emotion}')
            return True, response, emotion
        except:
            response = ""

        if "GOODBYE:" in response:
            response_text = self._clean_response(response.split("GOODBYE:")[-1])
            save_str = f"{self.name} 观测到 {observation} ,回复了 {response_text}"
            # 存 context
            self.context.add_context(save_str)
            # 存记忆
            self.memory.save_context(
                {},
                {self.memory.add_memory_key: save_str},
            )
            return False, f"{response_text}", ""
        regex_str = f"(SAY|{self.name}|{self.name} ?可能会回复|{self.name} ?会回复|{self.name} ?回复道|{self.name.upper()}" \
                    f"|{self.name.upper()} ?可能会回复|{self.name.upper()} ?会回复|{self.name.upper()} ?回复道)(:|：)"
        if re.search(regex_str, response):
            response_text = self._clean_response(re.split(regex_str, response)[-1])
            save_str = f"{self.name} 观测到 {observation}, 回复了 {response_text}"
            # 存 context
            self.context.add_context(save_str)
            # 存记忆
            self.memory.save_context(
                {},
                {self.memory.add_memory_key: save_str},
            )
            return True, f"{response_text}", ""
        else:
            save_str = f"{self.name} 观测到 {observation}, {response}"
            # 存 context
            self.context.add_context(save_str)
            # 存记忆
            self.memory.save_context(
                {},
                {self.memory.add_memory_key: save_str},
            )
            return True, response, ""

    def get_agent_description(self) -> str:
        """Return a description of the agent."""
        age = self.age if self.age is not None else "N/A"
        return f"名字: {self.name} (年龄: {age})\n特征: {self.traits}\n{self.relation}"
