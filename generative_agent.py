import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from generative_agent_memory import GenerativeAgentMemory
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
    status: str
    """The traits of the character you wish not to change."""
    memory: GenerativeAgentMemory
    """The memory object that combines relevance, recency, and 'importance'."""
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
    """Summary of the events in the plan that the agent took."""

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

    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "{observation}\n从以上观察中提取实体."
            + "\nEntity="
        )
        return self.chain(prompt).run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "{observation}\n基于上面给出的观察到的内容, 判断 {entity} 在做什么?"
            + "\n{entity} 正在"
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )

    def _generate_reaction(self, observation: str, suffix: str) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_description}"
            + "\n当前时间是: {current_time}."
            + "\n{agent_name} 的状态包括: {agent_status}"
            # + "\n{agent_name} 的相关上下文是:\n{relevant_memories}"  # 获取 agent 与 observation 中实体的关系
            + "\n相关记忆包括: {relevant_memories}"  # chain.run.prep_inputs 中注入 recent_memories, 但是粒度太粗,基本是所有上下文
            + "\n当前观测到的对话内容是: {observation}"
            + "\n\n"
            + suffix
        )
        agent_description = self.get_agent_description()  # 名字: {self.name} (年龄: {age})\n属性: {self.traits}\n{self.summary}
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs: Dict[str, Any] = dict(
            agent_description=agent_description,
            current_time=current_time_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
        )
        return self.chain(prompt=prompt).run(queries=[observation], **kwargs).strip()

    def _clean_response(self, text: str) -> str:
        return re.sub(f"^{self.name} ", "", text.strip()).strip()

    def generate_reaction(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            + " what would be an appropriate reaction? Respond in one line."
            + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split("\n")[0]
        # AAA
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}"
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = self._clean_response(result.split("SAY:")[-1])
            return True, f"{said_value}"
        else:
            return False, result

    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "如果要结束对话(告别), 就回复:GOODBYE: \"你要回复的内容\". 如果要继续对话, 就回复: SAY: \"你要回复的内容\"\n\n"
            "基于以上信息, {agent_name} 会回复什么内容? "
        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split("\n")[0]
        print(f"----dialogue_response----\n{result}")
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} 观测到 "
                    f"{observation} ,回复了 {farewell}"
                },
            )
            return False, f"{farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {response_text}"
                },
            )
            return True, f"{response_text}"
        else:
            return True, result

    def get_agent_description(self) -> str:
        """Return a descriptive summary of the agent."""
        age = self.age if self.age is not None else "N/A"
        return f"名字: {self.name} (年龄: {age})\n属性: {self.traits}"

    def get_full_header(self, force_refresh: bool = False) -> str:
        """Return a full header of the agent's status, summary, and current time."""
        description = self.get_agent_description()
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        print(f"----YANCY----get_full_header----self.summary\n{self.summary}")
        return (
            f"{description}\n当前时间是: {current_time_str}.\n{self.name} 的状态是: {self.status}"
        )
