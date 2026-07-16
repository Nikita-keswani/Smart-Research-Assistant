"""
Step 5: Evaluation

Score generated answers using RAGAS metrics:
- Faithfulness: is the answer grounded in the retrieved context?
- Answer Relevance: does the answer actually address the question?
- Context Precision: how relevant is the retrieved context to the question?

Wrapped in an `AnswerEvaluator` class so the metric list lives in one
place (encapsulation) and can be swapped/extended without touching
callers -- they just call `.evaluate(question, answer, source_documents)`.
"""
from typing import List

# Shim to fix Ragas/Langchain compatibility issue where it tries to import removed ChatVertexAI
import sys
import types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI: pass
    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

class AnswerEvaluator:
    """Runs RAGAS evaluation on a single Q&A pair."""

    def evaluate(self, question: str, answer: str, contexts: List[str]) -> dict:
        """
        Args:
            question: the user's question
            answer: the generated answer
            contexts: list of retrieved context strings (page_content of source docs)

        Returns:
            dict of metric_name -> score (0-1, higher is better)
        """
        print("[Evaluation] Setting up evaluation dataset...")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        
        metrics = [faithfulness, answer_relevancy, context_precision]

        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "reference": [answer]
            }
        )
        
        from config import config
        from langchain_openai import AzureChatOpenAI
        from src.embeddings import get_embedding_model

        print("[Evaluation] Initializing evaluator LLM and embedding models...")
        eval_llm = AzureChatOpenAI(
            azure_deployment=config.azure_openai_language_model,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_key,
            temperature=0,
        )
        eval_embeddings = get_embedding_model()

        print("[Evaluation] Running evaluation on dataset (this might take a few seconds)...")
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
        )
        scores = result.to_pandas().iloc[0].to_dict()
        print(f"[Evaluation] Evaluation scores: {scores}")
        return scores

    @staticmethod
    def contexts_from_source_documents(source_documents) -> List[str]:
        """Helper to turn LangChain source_documents into plain text list for RAGAS."""
        return [doc.page_content for doc in source_documents]
