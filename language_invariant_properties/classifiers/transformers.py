from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from language_invariant_properties.classifiers.datasets import MainDataset
from transformers import AdamW
from language_invariant_properties.abstraction import AbstractClassifier
from torch.utils.data import DataLoader
import os
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

LANGUAGE_MAPPER = \
{
    "english": ("roberta-base", "roberta-base"),
    "italian": ("Musixmatch/umberto-commoncrawl-cased-v1", "Musixmatch/umberto-commoncrawl-cased-v1")
}


class TransformerClassifier(AbstractClassifier):

    def __init__(self, language, experiment_name = None):
        super().__init__()
        if language not in LANGUAGE_MAPPER.keys():
            raise Exception("Language not supported")

        self.language = language
        self.tuner = None
        self.label_encoder = LabelEncoder()
        self.experiment_name = experiment_name + "_" + language

    def train(self, text, labels):
        model_name, tokenizer_name = LANGUAGE_MAPPER[self.language]

        labels = self.label_encoder.fit_transform(labels)

        self.tuner = TransformerTuner(model_name, tokenizer_name)
        final_f1 = self.tuner.train_with_es(text, labels)

    def predict(self, text):
        return self.label_encoder.inverse_transform(self.tuner.predict(text))


class TransformerTuner:

    def __init__(self, model_name, tokenizer_name):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = None
        self.tokenizer = None

    def train_with_es(self, texts, labels, epochs=10, batch_size=16, learning_rate=5e-5, save_name=None):
        """
        This methods train the transformer using early stopping on the MacroF1.
        """

        config = AutoConfig.from_pretrained(self.model_name, num_labels=len(set(labels)),
                                            finetuning_task="custom")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        texts = np.array(texts)
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.15, random_state=11)

        current_f1 = -1

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)

        # not the smartest way to do this, but faster to code up
        tokenized_train = tokenizer(X_train, truncation=True, padding=True)
        tokenized_test = tokenizer(X_test, truncation=True, padding=True)

        train_dataset = MainDataset(tokenized_train, y_train)
        test_dataset = MainDataset(tokenized_test, y_test)

        model.to(self.device)
        model.train()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        optim = AdamW(model.parameters(), lr=learning_rate)

        pbar = tqdm(total=epochs, position=0, leave=True)
        for epoch in range(epochs):
            pbar.update(1)
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                lab = batch['labels'].to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=lab)

                loss = outputs[0]
                loss.backward()
                optim.step()

            predicted = []
            with torch.no_grad():
                model.eval()
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    predicted.extend(torch.argmax(outputs["logits"], axis=1).cpu().numpy().tolist())
                model.train()

            new_f1 = f1_score(y_test, predicted)

            if current_f1 < new_f1:
                current_f1 = new_f1
            else:
                break

        pbar.close()

        self.model = model
        self.tokenizer = tokenizer

        if save_name is not None:
            os.makedirs(save_name)
            model.save_pretrained(save_name)
            tokenizer.save_pretrained(save_name)

        return current_f1

    def predict(self, texts, batch_size):


        tokenized_train = self.tokenizer(texts, truncation=True, padding=True)

        labels = [0] * len(texts)  # I know... I know

        train_dataset = MainDataset(tokenized_train, labels)

        self.model.to(self.device)
        self.model.eval()

        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions.extend(torch.argmax(outputs["logits"], axis=1).cpu().numpy().tolist())

        return predictions
