"""
Retriever that combines embedding similarity with recency in retrieving values.
TODO 修改 memory stream 逻辑，聊天记录和 memory 拆分
"""
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.base import VectorStore


def _get_hours_passed(time: datetime, ref_time: datetime) -> float:
    """Get the hours passed between two datetime objects."""
    return (time - ref_time).total_seconds() / 3600


class TimeWeightedVectorStoreRetriever(BaseRetriever, BaseModel):
    """Retriever combining embededing similarity with recency."""

    vectorstore: VectorStore
    """The vectorstore to store documents and determine salience."""

    search_kwargs: dict = Field(default_factory=lambda: dict(k=100))
    """Keyword arguments to pass to the vectorstore similarity search."""

    # TODO: abstract as a queue
    # 作为消息快照的存储，跟原本的 memory 解耦
    memory_stream: List[Document] = Field(default_factory=list)
    """The memory_stream of documents to search through."""

    decay_rate: float = Field(default=0.01)
    """The exponential decay factor used as (1.0-decay_rate)**(hrs_passed)."""

    k: int = 4
    """The maximum number of documents to retrieve in a given call."""

    other_score_keys: List[str] = []
    """Other keys in the metadata to factor into the score, e.g. 'importance'."""

    default_salience: Optional[float] = None
    """The salience to assign memories not retrieved from the vector store.

    None assigns no salience to documents not fetched from the vector store.
    """

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: Optional[float],
        current_time: datetime,
    ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            document.metadata["last_accessed_at"],
        )
        score = (1.0 - self.decay_rate) ** hours_passed
        for key in self.other_score_keys:
            if key in document.metadata:
                score += document.metadata[key]
        if vector_relevance is not None:
            score += vector_relevance
        return score

    def get_salient_docs(self, query: str) -> Dict[int, Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        docs_and_scores: List[Tuple[Document, float]]
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, **self.search_kwargs
        )
        results = {}
        # TODO 原代码里的 buffer id 怀疑是想记录现在内存 or redis 里是否存了聊天，从而从 memory  stream 里拿
        # TODO 即 memory-stream 的内容 buffer-id 是一一对应的，但是目前还没看出来有更多的作用
        for fetched_doc, relevance in docs_and_scores:
            if "buffer_idx" in fetched_doc.metadata:
                buffer_idx = fetched_doc.metadata["buffer_idx"]
                # doc = self.memory_stream[buffer_idx]
                doc = fetched_doc
                results[buffer_idx] = (doc, relevance)
        return results

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Return documents that are relevant to the query.
        """
        current_time = datetime.now()
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k :]
        }
        # If a doc is considered salient, update the salience score
        docs_and_scores.update(self.get_salient_docs(query))
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        # Ensure frequently accessed memories aren't forgotten
        current_time = datetime.now()
        for doc, _ in rescored_docs[: self.k]:
            # TODO: Update vector store doc once `update` method is exposed.
            buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["last_accessed_at"] = current_time
            result.append(buffered_doc)
        return result

    def load_memories_from_local(
            self, folder_path: str, embeddings: Embeddings, index_name: str = "index"
    ):
        """
        从本地 fassi 加载 memory
        TODO 定制方法，仅针对 faiss （用于加载记忆调试 memory 逻辑，后面替换成 milvus 后可删除）
        :return:
        """
        self.vectorstore = self.vectorstore.load_local(
            folder_path, embeddings, index_name
        )
        # 按照现在的逻辑需要把所有的内容都加入 memory stream 中
        memory_stream = {}
        for index, doc_id in self.vectorstore.index_to_docstore_id.items():
            doc = self.vectorstore.docstore.search(doc_id)
            memory_stream[doc.metadata['buffer_idx']] = doc
        # memory stream 中存的是list 顺序是按照 buffer id 来的
        # 要考虑读取时候顺序被打乱的情况（做排序）
        memory_stream = sorted(memory_stream.items(), key=lambda x:x[0])
        self.memory_stream = [x[1] for x in memory_stream]

    def save_memories_to_local(self, folder_path: str):
        '''
        把记忆存在本地
        TODO 定制方法，仅针对 faiss （用于加载记忆调试 memory 逻辑，后面替换成 milvus 后可删除）
        :return:
        '''
        self.vectorstore.save_local(folder_path)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Return documents that are relevant to the query."""
        raise NotImplementedError

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time", datetime.now())
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time", datetime.now())
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return await self.vectorstore.aadd_documents(dup_docs, **kwargs)
