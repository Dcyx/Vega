"""
Retriever that combines embedding similarity with recency in retrieving values.
修改方案：
memory_stream 和 vectorstore 解耦
- memory_stream 存聊天上文，在 retriever 里删除
- vectorstore 存聊天上下文 and 一些 reflection，实时写 & 实时读

"""
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.base import VectorStore


def _get_hours_passed(time: float, ref_time: float) -> float:
    """Get the hours passed between two datetime objects."""
    return (time - ref_time) / 3600


class TimeWeightedVectorStoreRetriever(BaseRetriever, BaseModel):
    """Retriever combining embededing similarity with recency."""

    vectorstore: VectorStore
    """The vectorstore to store documents and determine salience."""

    search_kwargs: dict = Field(default_factory=lambda: dict(k=100))
    """Keyword arguments to pass to the vectorstore similarity search."""

    # TODO: abstract as a queue
    # 2023/5/11 删除 memory_stream 相关内容，消息记录用其他方式进行存储
    # memory_stream: List[Document] = Field(default_factory=list)
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
        """
        Return the combined score for a document.
        根据时间间隔计算 score 衰减情况
        """
        hours_passed = _get_hours_passed(
            current_time.timestamp(),
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
        """
        Return documents that are salient to the query.
        从向量库中检索与 query 最相似的内容
        """
        docs_and_scores: List[Tuple[Document, float]]
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, **self.search_kwargs
        )
        results = {}
        for fetched_doc, relevance in docs_and_scores:
            result_index = len(results)
            results[result_index] = (fetched_doc, relevance)
        return results

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Return documents that are relevant to the query.
        根据向量相似度找到最相似 k 个doc 返回
        """
        current_time = datetime.now()
        docs_and_scores = self.get_salient_docs(query)
        # 根据时间间隔计算相关性得分 score 的衰减情况（如果该doc 还包含 importance 得分之类的，则要加进去
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs[: self.k]:
            doc.metadata["last_accessed_at"] = current_time.timestamp()
            result.append(doc)
        return result

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Return documents that are relevant to the query."""
        raise NotImplementedError

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """
        Add documents to vectorstore.
        """
        print(f"\t>> [retriever] add_documents documents = {documents}")
        current_time = kwargs.get("current_time", datetime.now())
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        # 更新记忆的访问时间
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time.timestamp()
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time.timestamp()
            # 删除 buffer_idx 将来和 context 统一设置后，从上一层传进来
            # doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        # 2023/5/11 解耦 memory stream
        # self.memory_stream.extend(dup_docs)
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
                doc.metadata["last_accessed_at"] = current_time.timestamp()
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time.timestamp()
        return await self.vectorstore.aadd_documents(dup_docs, **kwargs)

    def delete_document_by_primary_keys(self, primary_keys: List[int], **kwargs: Any):
        """
        Delete document by primary keys.
        """
        return self.vectorstore.delete_by_primary_keys(primary_keys, **kwargs)
