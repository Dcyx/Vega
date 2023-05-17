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
            "基础信息: {agent_description} "
            + "\n当前时间是: {current_time}."
            + "\n当前场景下对 {agent_name} 的要求包括: {agent_claim}"
            + "\n相关记忆: {relevant_memories}"  # chain.run.prep_inputs 中检索 relevant_memories
            + "\n聊天记录：{last_context}"
            + "\n\n回复格式: 如果回复内容中包含告别的语义, 例如 拜拜、再见、下次再聊等, 就回复 'GOODBYE:\"回复内容\"'; 如果要继续对话, 就回复 'SAY:\"回复内容\"'. "
              "不要直接将背景信息作为回复,回答的简洁一点,不要啰嗦,注意对话上下文."
            + "\n\n基于观测到的聊天信息 '{observation}', {agent_name} 会回复什么内容?"
        )
        agent_description = self.get_agent_description()
        agent_claim, obs_summary = self.get_agent_claim(agent_description=agent_description, observation=observation)
        last_context = self.get_context()
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs: Dict[str, Any] = dict(
            agent_description=agent_description,
            current_time=current_time_str,
            agent_name=self.name,
            agent_claim=agent_claim,
            last_context=last_context,
            observation=observation,
        )
        response = self.chain(prompt=prompt).run(queries=[observation, obs_summary], **kwargs).strip()
        print(f"----Yancy----\n{response}\n")
        return response

    def get_agent_claim(self, agent_description, observation: str) -> str:
        """
        对当前情境进行抽象. 检索相关记忆并提炼 agent_claim, 返回 claim 和 observation 的抽象
        """
        # TODO:  backward,forward
        prompt_obs_summary = PromptTemplate.from_template(
            "历史聊天记录:"
            "{last_context}"
            "最新输入:"
            "\n- {time_str}\t{observation}"
            "\n基于以上对话内容,对消息中的场景、人物关系、事件等进行抽象, 总结成一句话. 不要过程,直接给结论."
            "\n\n"
        )
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        obs_summary = self.chain(prompt_obs_summary).run(
            observation=observation, last_context=self.get_context(), time_str=current_time_str
        ).strip()
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

    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        response_raw = self._generate_reaction(observation)
        response = response_raw.strip().split("\n")[0]

        if "GOODBYE:" in response:
            response_text = self._clean_response(response.split("GOODBYE:")[-1])
            save_str = f"{observation}。{self.name} 回复了 {response_text}"
            # 存 context
            self.context.add_context(save_str)
            # 存记忆
            self.memory.save_context(
                {},
                {self.memory.add_memory_key: save_str},
            )
            return False, f"{response_text}"
        regex_str = f"(SAY|{self.name}|{self.name} ?可能会回复|{self.name} ?会回复|{self.name} ?回复道|{self.name.upper()}" \
                    f"|{self.name.upper()} ?可能会回复|{self.name.upper()} ?会回复|{self.name.upper()} ?回复道)(:|：)"
        if re.search(regex_str, response):
            response_text = self._clean_response(re.split(regex_str, response)[-1])
            save_str = f"{observation}。{self.name} 回复了 {response_text}"
            # 存 context
            self.context.add_context(save_str)
            # 存记忆
            self.memory.save_context(
                {},
                {self.memory.add_memory_key: save_str},
            )
            return True, f"{response_text}"
        else:
            save_str = f"{observation}。{self.name} 回复了 {response}"
            # 存 context
            self.context.add_context(save_str)
            # 存记忆
            self.memory.save_context(
                {},
                {self.memory.add_memory_key: save_str},
            )
            return True, response

    def get_agent_description(self) -> str:
        """Return a description of the agent."""
        age = self.age if self.age is not None else "N/A"
        return f"名字: {self.name} (年龄: {age})\n特征: {self.traits}{self.relation}"
