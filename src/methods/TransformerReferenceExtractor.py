import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TransformerReferenceExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.5):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def match_query(self, context, query):
        context_embeddings = self.model.encode(context)
        query_embedding = self.model.encode([query])

        # Calculate cosine similarity
        similarities = np.dot(context_embeddings, query_embedding.T).flatten()

        # Convert to one-hot encoding
        one_hot = (similarities > self.threshold).astype(int)

        return one_hot, similarities

    # def find_best_threshold(self, contexts, queries, labels):
    #     all_similarities = []
    #     all_true_labels = []
    #
    #     for context, query, label in zip(contexts, queries, labels):
    #         _, similarities = self.match_query(context, query)
    #         all_similarities.extend(similarities)
    #         all_true_labels.extend(label)
    #
    #     thresholds = np.arange(0.1, 1.0, 0.05)
    #     best_accuracy = 0
    #     best_threshold = self.threshold
    #
    #     for threshold in thresholds:
    #         predictions = (np.array(all_similarities) > threshold).astype(int)
    #         accuracy = np.mean(predictions == all_true_labels)
    #
    #         if accuracy > best_accuracy:
    #             best_accuracy = accuracy
    #             best_threshold = threshold
    #
    #     self.threshold = best_threshold
    #     return best_threshold, best_accuracy


def test_extractor():
    contexts = [
        ["46", "47", "the american convention", "30", "37"],
        ["46", "47", "the american convention", "30", "37"],
        ["46", "47", "the american convention", "30", "37"],
    ]

    queries = [
        "46 of the american convention",
        "30 of the convention",
        "47 of document",
        "37 of document",
    ]

    labels = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]

    extractor = TransformerReferenceExtractor()

    # best_threshold, best_accuracy = extractor.find_best_threshold(contexts, queries, labels)
    # print(f"Best threshold: {best_threshold:.2f}")
    # print(f"Best accuracy: {best_accuracy:.2f}")

    # Test some queries
    test_cases = [
        {"context": ["46", "47", "the american convention", "30", "37"], "query": "46 of the american convention"},
        {"context": ["46", "47", "the american convention", "30", "37"], "query": "47 of the american convention"},
        {"context": ["4", "5", "8", "1", "17"], "query": "5 of the convention"},
    ]

    print("\nTest Results:")
    print("-" * 50)

    for test in test_cases:
        one_hot, similarities = extractor.match_query(test["context"], test["query"])

        print(f"\nContext: {test['context']}")
        print(f"Query: {test['query']}")
        print(f"One-hot encoding: {one_hot}")
        print(f"Similarities: {similarities.round(3)}")


if __name__ == "__main__":
    model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16
    )
    model.eval()

    pairs = [
        ["46 47 the american convention 30 37", "46 of the american convention"],
        ["46 47 the american convention 30 37", "47 of the american convention"],
        ["46 47 the american convention 30 37", "30 of the american convention"],
        ["46 47 the american convention 30 37", "37 of the american convention"],
        ["46 47 the american convention 30 37", "28 of the american convention"],
        ["46 47 the american convention 30 37", "47 of the 37"],
        ["46 47 the american convention 30 37", "37 of the 47"],
        ["46 47 the american convention 30 37", "46 of the 47"],
    ]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        scores = (
            model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )
        print(scores)
    # test_extractor()
