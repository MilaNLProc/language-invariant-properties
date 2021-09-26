from sklearn.preprocessing import LabelEncoder
from language_invariant_properties.metrics import *
from language_invariant_properties.classifiers.transformers import TransformerClassifier
from language_invariant_properties.classifiers.sklearn_wrapper import LogisticRegressionClassifier, LogisticRegressionSTClassifier
import abc
from language_invariant_properties.abstractions.classifiers import AbstractClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging


class Dataset(abc.ABC):

    def __init__(self, source_language, target_language, dataset_name,
                 source_classifier: AbstractClassifier = None,
                 target_classifier: AbstractClassifier = None,
                 transformer=False, sentence_embedding=False, common_classifier=False):
        self.source_classifier = source_classifier
        self.target_classifier = target_classifier
        self.source_language = source_language
        self.target_language = target_language
        self.source_predictions = None
        self.target_predictions = None
        self.original_data = None
        self.transformer = transformer
        self.sentence_embedding = sentence_embedding
        self.dataset_name = dataset_name
        self.common_classifier = common_classifier

    @abc.abstractmethod
    def get_text_to_translate(self):
        pass

    @abc.abstractmethod
    def train_data(self):
        pass

    def score(self, translated_language_data):
        source_train, target_train, original_language_data = self.train_data()

        prediction_on_original, prediction_on_transformed = self.compute(original_language_data,
                                                                          translated_language_data,
                                                                          source_train, target_train)

        original_data = original_language_data["property"].values.tolist()

        self.source_predictions = prediction_on_original
        self.target_predictions = prediction_on_transformed
        self.original_data = original_data

        def normalize_counter(dictionary):
            dictionary = dict(dictionary)
            summed = sum(list(dictionary.values()))
            for k in dictionary:
                dictionary[k] = round(dictionary[k]/summed, 2)
            return dictionary

        return {"KL_O_PO": get_kl(original_data, prediction_on_original),
                "KL_O_PT":  get_kl(original_data, prediction_on_transformed),
                "significance_source": get_significance(original_data, prediction_on_original),
                "significance_transformed": get_significance(original_data, prediction_on_transformed),
                "distribution_O": normalize_counter(Counter(original_data)),
                "distribution_PO": normalize_counter(Counter(prediction_on_original)),
                "distribution_PT": normalize_counter(Counter(prediction_on_transformed)),
                }

    def train_classifier(self, text, labels, language):

        if self.transformer:
            logging.info("Training Transformers: " + language)
            tc = TransformerClassifier(language, "LIPs_" + str(language) + "_" + self.dataset_name)
            tc.train(text, labels)
            return tc

        elif self.sentence_embedding:

            if self.common_classifier and self.source_language == "english":
                language_passing = language
            else:
                language_passing = None

            logging.info("Training Logistic Regression Classifier with MultiLingual Sentence Transformer: " + language)
            lc = LogisticRegressionSTClassifier(language=language_passing)
            lc.train(text, labels)
            return lc

        else:
            logging.info("Training Logistic Regression Classifier: " + language)
            clf = LogisticRegressionClassifier()
            clf.train(text, labels)
            return clf

    def plot(self, save_path):
        df = pd.DataFrame()

        data = list(map(lambda x: dict(Counter(list(map(str, x)))),
                        [self.original_data, self.target_predictions, self.source_predictions]))

        for name, dictt in zip("O,PT,PO".split(","),
                               data):
            for _class in dictt:
                for value in range(0, dictt[_class]):
                    df = df.append({"Type": name, "Property": _class}, ignore_index=True)

        sns.histplot(data=df, x="Property", hue="Type", multiple="dodge", shrink=.8)


        plt.savefig(save_path)
        plt.show()

    def compute(self, original_language_data, translated_language_data, source_train=None, target_train=None):

        assert len(original_language_data["text"].values) == len(translated_language_data)

        if source_train is not None and target_train is not None:

            le = LabelEncoder()

            source_labels = le.fit_transform(source_train["property"].values)
            target_labels = le.transform(target_train["property"].values)

            source_classifier = self.train_classifier(source_train["text"].values, source_labels, self.source_language)
            source_predictions = source_classifier.predict(original_language_data["text"].values)
            if self.common_classifier:
                target_predictions = source_classifier.predict(translated_language_data)
            else:
                target_classifier = self.train_classifier(target_train["text"].values, target_labels, self.target_language)
                target_predictions = target_classifier.predict(translated_language_data)

            return le.inverse_transform(source_predictions), le.inverse_transform(target_predictions)

        else:

            if self.source_classifier is None and self.target_classifier is None:
                raise Exception("Source and Target Classifiers not found")

            return self.source_classifier.predict(original_language_data), \
                   self.target_classifier.predict(translated_language_data)

