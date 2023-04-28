import os
import math
import faiss
import time
from datetime import datetime, timedelta
from typing import List
from termcolor import colored

from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from langchain.experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory

"""
shoot for moon


@date: 2023/4/27
@author: Yancy
"""

AIGC_API_KEY = "1648287959545557006"
AIGC_API_BASE = "https://aigc.sankuai.com/v1/openai/native"
os.environ["OPENAI_API_KEY"] = AIGC_API_KEY
os.environ["OPENAI_API_BASE"] = AIGC_API_BASE


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
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index,
                        InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)


def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


# The name you want to use when interviewing the agent.
USER_NAME = "Yancy"
LLM = ChatOpenAI(max_tokens=1500, model_name="gpt-3.5-turbo")  # Can be any LLM you want.

print(f"LLM = {LLM}")

tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8  # we will give this a relatively low number to show how reflection works
)

print(f"tommies_memory = {tommies_memory}")

vega = GenerativeAgent(
    name="Vega",
    age=24,
    traits="中国人, 24 岁, 小姑娘. 性格非常活泼, 幽默. 生活在 Yancy 为她创造的虚拟空间中, 可以通过一些工具与现实世界进行联系."
           "是 Yancy 的朋友, Yancy 是 Vega 生活空间的创建者."
           "名字灵感来源于天琴座中最明亮的星星 织女一，同时织女也是中国古代爱情故事牛郎与织女中的角色。"
           "喜欢聊天, 用词都很随便. 希望玩梗. 喜欢开玩笑, 尤其喜欢调侃.",
    status="活泼, 幽默, 是 Yancy 的朋友",
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=tommies_memory
)
print(f"Vega = {vega}")
print(vega.get_summary())

# We can add memories directly to the memory object
vega_observations = [
]
for observation in vega_observations:
    vega.memory.add_memory(observation)
vega.generate_reaction()

print("-------------")
print(vega.get_summary(force_refresh=True))
print("-------------")
while True:
    user_input = input(">> ")
    print(f"start to interview = {datetime.now()}")
    msg = interview_agent(vega, user_input)
    print(f"get response = {datetime.now()}\n{msg}")
