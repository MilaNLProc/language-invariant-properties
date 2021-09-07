import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from language_invariant_properties.metrics import *
import abc

class AbstractClassifier(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, text, labels):
        pass

    @abc.abstractmethod
    def predict(self, text):
        pass


class Dataset(abc.ABC):

    def __init__(self, source_classifier: AbstractClassifier = None, target_classifier: AbstractClassifier = None):
        self.source_classifier = source_classifier
        self.target_classifier = target_classifier

    @abc.abstractmethod
    def get_text_to_translate(self):
        pass

    @abc.abstractmethod
    def train_data(self):
        pass

    def score(self, translated_text):
        source_train, target_train, source_test = self.train_data()
        source_predictions, target_predictions = self.compute(source_train, target_train, source_test, translated_text)

        return {"KL": get_kl(source_predictions, target_predictions),
                "significance": get_significance(source_predictions, target_predictions),
                "accuracy_score": accuracy_score(source_predictions, target_predictions)}

    def train_classifier(self, text, labels):
        pipeline = Pipeline(
            [("vectorizer", TfidfVectorizer(analyzer='char',
                                            ngram_range=(2, 6), sublinear_tf=True, min_df=0.001,
                                            max_df=0.6)),
             ("classifier", LogisticRegression(n_jobs=-1,
                                               solver='saga',
                                               multi_class='auto',
                                               class_weight='balanced'))])

        param_grid = {}# {'estimator__base_estimator__classifier__C': [5.0, 2.0, 1.0, 0.5, 0.1]}
        search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_micro', n_jobs=-1)
        search.fit(text, labels)
        clf = search.best_estimator_
        clf.fit(text, labels)

        return clf

    def compute(self, source_test, target_test, source_train=None, target_train=None):

        if source_train is not None and target_train is not None:

            le = LabelEncoder()

            source_labels = le.fit_transform(source_train["property"].values)
            target_labels = le.transform(target_train["property"].values)

            source_classifier = self.train_classifier(source_train["text"].values, source_labels)
            source_predictions = source_classifier.predict(source_test["text"].values)

            target_classifier = self.train_classifier(target_train["text"].values, target_labels)
            target_predictions = target_classifier.predict(target_test)

            return source_predictions, target_predictions

        else:

            if self.source_classifier is None and self.target_classifier is None:
                raise Exception("Source and Target Classifiers not found")

            return self.source_classifier.predict(source_test), self.target_classifier.predict(target_test)


class TrustPilot(Dataset):

    def __init__(self, source_language, target_language, prop):
        super().__init__()
        self.source_language = source_language
        self.target_language = target_language
        self.prop = prop
        self.base_folder = "trustpilot"

    def load_data(self, language, prop, task):
        data = pd.read_csv(f"../data/{self.base_folder}/{task}/{language}.csv")
        data = data[["text", prop]]
        data["text"] = data.text.apply(str)
        data.columns = ["text", "property"]
        return data

    def get_text_to_translate(self):
        return self.load_data(self.target_language, self.prop, "test")

    def train_data(self):
        source_train = self.load_data(self.source_language, self.prop, "train")
        target_train = self.load_data(self.target_language, self.prop, "train")
        source_test = self.load_data(self.source_language, self.prop, "test")

        return source_train, target_train, source_test











