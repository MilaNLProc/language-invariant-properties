from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
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

    def score(self, translated_language_data):
        source_train, target_train, original_language_data = self.train_data()

        source_predictions, target_predictions = self.compute(original_language_data, translated_language_data,
                                                              source_train, target_train)

        original_data = original_language_data["property"].values.tolist()

        return {"KL_source": get_kl(original_data, source_predictions),
                "KL_transformed":  get_kl(original_data, target_predictions),
                #"significance_source": get_significance(original_data, source_predictions),
                #"significance_transformed": get_significance(original_data, target_predictions),
                "original_distribution": dict(Counter(original_data)),
                "source_distribution": dict(Counter(source_predictions)),
                "transformed_distribution": dict(Counter(target_predictions)),
                }

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

    def compute(self, original_language_data, translated_language_data, source_train=None, target_train=None):

        assert len(original_language_data["text"].values) == len(translated_language_data)

        if source_train is not None and target_train is not None:

            le = LabelEncoder()

            source_labels = le.fit_transform(source_train["property"].values)
            target_labels = le.transform(target_train["property"].values)

            source_classifier = self.train_classifier(source_train["text"].values, source_labels)
            source_predictions = source_classifier.predict(original_language_data["text"].values)

            target_classifier = self.train_classifier(target_train["text"].values, target_labels)
            target_predictions = target_classifier.predict(translated_language_data)

            return le.inverse_transform(source_predictions), le.inverse_transform(target_predictions)

        else:

            if self.source_classifier is None and self.target_classifier is None:
                raise Exception("Source and Target Classifiers not found")

            return self.source_classifier.predict(original_language_data), \
                   self.target_classifier.predict(translated_language_data)

