from language_invariant_properties.abstractions.classifiers import AbstractClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class LogisticRegressionClassifier(AbstractClassifier):

    def __init__(self):
        super().__init__()
        self.label_encoder = LabelEncoder()

    def train(self, text, labels):

        labels = self.label_encoder.fit_transform(labels)

        pipeline = Pipeline(
            [("vectorizer", TfidfVectorizer(analyzer='char',
                                            ngram_range=(2, 6), sublinear_tf=True, min_df=0.001,
                                            max_df=0.6)),
             ("classifier", LogisticRegression(n_jobs=-1,
                                               solver='saga',
                                               multi_class='auto',
                                               class_weight='balanced'))])

        param_grid = {'classifier__C': [5.0, 2.0, 1.0, 0.5, 0.1]}
        search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_micro', n_jobs=-1)
        search.fit(text, labels)
        self.model = search.best_estimator_
        self.model.fit(text, labels)

    def predict(self, text):
        return self.label_encoder.inverse_transform(self.model.predict(text))
