import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class LightGBMReferenceExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
        self.label_encoder = LabelEncoder()
        self.model = None

    def train_tfidf(self, contexts, queries):
        all_texts = []
        for context in contexts:
            all_texts.extend([str(item) for item in context])
        print("all texts before: " + str(all_texts))
        all_texts.extend(queries)
        print("all texts after: " + str(all_texts))
        self.tfidf.fit(all_texts)

    def create_features(self, context_item, query):
        print(f"CONTEXT ITEM: {context_item}")
        print(f"QUERY: {query}")
        exact_match = int(str(context_item).lower() == query.lower())
        print(f"EXACT MATCH: {exact_match}")

        try:
            context_num = float(str(context_item))
            query_num = float("".join(filter(str.isdigit, query)))
            num_diff = abs(context_num - query_num)
            has_numbers = 1
            print(f"{context_num=} || {query_num=} || {num_diff=} || {has_numbers=}")

        except ValueError:
            num_diff = -1
            has_numbers = 0

        len_diff = abs(len(str(context_item)) - len(query))

        combined_text = f"{context_item} {query}"
        tfidf_features = self.tfidf.transform([combined_text]).toarray()

        # features = np.hstack([
        #     [exact_match, num_diff, has_numbers, len_diff],
        #     tfidf_features[0]
        # ])

        features = tfidf_features[0]
        return features

    def prepare_training_data(self, contexts, queries, labels):
        all_features = []
        all_labels = []

        for context, query, label in zip(contexts, queries, labels):
            context_features = []
            for context_item in context:
                features = self.create_features(context_item, query)
                context_features.append(features)

            all_features.extend(context_features)
            all_labels.extend(label)

        return np.array(all_features), np.array(all_labels)

    def train(self, contexts, queries, labels):
        self.train_tfidf(contexts, queries)

        X, y = self.prepare_training_data(contexts, queries, labels)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
        }

        train_data = lgb.Dataset(X, label=y)

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=2,
            valid_sets=[train_data],
            # early_stopping_round=10,
            # verbose_eval=10
        )

    def predict(self, context, query):

        features_list = []
        for context_item in context:
            features = self.create_features(context_item, query)
            features_list.append(features)

        features_list = np.array(features_list)
        predictions = self.model.predict(features_list)
        print("predictions: ", predictions)

        threshold = 0.5
        one_hot = (predictions > threshold).astype(int)

        return one_hot


def create_sample_data():
    contexts = [
        ["46", "47", "the american convention", "30", "37"],
        ["46", "47", "the american convention", "30", "37"],
        ["46", "47", "the american convention", "30", "37"],
        ["46", "47", "the american convention", "30", "37"],
    ]

    queries = [
        "46 of the american convention",
        "47 of the american convention",
        "30 of the american convention",
        "37 of the american convention",
    ]

    labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

    return contexts, queries, labels


def main():
    contexts, queries, labels = create_sample_data()

    extractor = LightGBMReferenceExtractor()
    extractor.train(contexts, queries, labels)

    test_context = ["46", "47", "the american convention", "30", "37"]
    test_query = "46 of the american convention"
    prediction = extractor.predict(test_context, query=test_query)

    print(f"Context: {test_context}")
    print(f"Query: {test_query}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
