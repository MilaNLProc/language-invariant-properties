import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from language_invariant_properties.metrics import *
import abc
from tqdm import tqdm

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
        source_predictions, target_predictions = self.compute(source_test, translated_text, source_train, target_train)

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

    def compute(self, source_test, target_test, source_train, target_train):
        assert len(source_test["text"].values) == len(target_test)

        if source_train is not None and target_train is not None:

            le = LabelEncoder()

            source_labels = le.fit_transform(source_train["property"].values)
            target_labels = le.transform(target_train["property"].values)

            source_classifier = self.train_classifier(source_train["text"].values, source_labels)
            source_predictions = source_classifier.predict(source_test["text"].values)

            target_classifier = self.train_classifier(target_train["text"].values, target_labels)
            target_predictions = target_classifier.predict(target_test)

            print(len(target_predictions, ))

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
        data = pd.read_csv(f"data/{self.base_folder}/{task}/{language}.csv")
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

class SemEval(Dataset):

    def __init__(self, source_language, target_language):
        super().__init__()
        self.source_language = source_language
        self.target_language = target_language
        self.base_folder = "semeval"
        self.text_to_translate = None

    def load_data(self, language, task):
        data = pd.read_csv(f"data/{self.base_folder}/{task}/{language}.csv")
        data = data[["text", "HS"]]
        data["text"] = data.text.apply(str)
        g = data.groupby('HS')
        data = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

        data.columns = ["text", "property"]
        return data

    def get_text_to_translate(self):
        if self.text_to_translate is None:
            self.text_to_translate = self.load_data(self.source_language, "test")
            return self.text_to_translate
        else:
            return self.text_to_translate

    def train_data(self):
        source_train = self.load_data(self.source_language, "train")
        target_train = self.load_data(self.target_language,  "train")
        source_test = self.get_text_to_translate()
        return source_train, target_train, source_test

tp = SemEval("spanish", "english")

k = (tp.get_text_to_translate()["text"].values)

from transformers import MarianTokenizer, MarianMTModel
from transformers import pipeline


mname = 'Helsinki-NLP/opus-mt-es-en'

model = MarianMTModel.from_pretrained(mname)
tok = MarianTokenizer.from_pretrained(mname)
translation = pipeline("translation_es_to_en", model=model, tokenizer=tok)

t = []
pbar = tqdm(total=len(k))
for a in k:
    t.append(translation(a)[0]["translation_text"])
    pbar.update(1)

print(tp.score(t))



