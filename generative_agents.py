import os
import configparser
import math
import faiss
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from generative_agent import GenerativeAgent
from generative_agent_memory import GenerativeAgentMemory

"""
shoot for moon


@date: 2023/4/27
@author: Yancy
"""

config_private = 'config_private.ini'
with open(config_private, mode='r', encoding='utf-8') as f:
    # 读取配置文件内容
    config = configparser.ConfigParser()
    config.read_file(f)
os.environ["OPENAI_API_KEY"] = config.get("OpenAI", "api_key")
os.environ["OPENAI_API_BASE"] = config.get("OpenAI", "api_base")
USER_NAME = config.get("User", "name")

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


# The name you want to use when interviewing the agent.
LLM = ChatOpenAI(max_tokens=1500, model_name="gpt-3.5-turbo")  # Can be any LLM you want.
print(f"LLM = {LLM}")

print("-------------")
vega_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8  # we will give this a relatively low number to show how reflection works
)
# vega_memory.pause_to_reflect()  #
print(f"vega_memory = {vega_memory}")

print("-------------")
vega = GenerativeAgent(
    name="Vega",
    age=24,
    traits="中国人, 24 岁, 小姑娘. 性格非常活泼, 幽默. 生活在 Yancy 为她创造的虚拟空间中, 可以通过一些工具与现实世界进行联系."
           "是 Yancy 的朋友, Yancy 是 Vega 生活空间的创建者."
           "名字灵感来源于天琴座中最明亮的星星 织女一，同时织女也是中国古代爱情故事牛郎与织女中的角色。"
           "喜欢聊天, 用词都很随便. 希望玩梗. 喜欢开玩笑, 尤其喜欢调侃.",
    status="活泼, 幽默, 是 Yancy 的朋友",
    llm=LLM,
    memory=vega_memory
)
print(f"Vega = {vega}")

# YANCY: 后续可用于启动时加载记忆?
# # We can add memories directly to the memory object
# vega_observations = [
# ]
# for observation in vega_observations:
#     vega.memory.add_memory(observation)

print("-------------")
while True:
    user_input = input(">> ")
    print(f"start to interview = {datetime.now()}")
    new_message = f"{USER_NAME} says {user_input}"
    msg = vega.generate_dialogue_response(new_message)[1]

    print(f"get response = {datetime.now()}\n{msg}")
