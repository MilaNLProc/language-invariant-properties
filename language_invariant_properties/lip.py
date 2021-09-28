import pandas as pd
from language_invariant_properties.abstractions.dataset import Dataset
import os
import re


class TrustPilotPara(Dataset):

    def __init__(self, source_language, target_language, prop, folder_path, **kwargs):
        super().__init__(source_language, target_language, common_classifier=True, **kwargs)

        self.prop = prop
        self.base_folder = folder_path

    def load_data(self, language, prop, task):
        data = pd.read_csv(f"{self.base_folder}/{task}/{language}.csv")

        data = data[["text", prop]]
        data["text"] = data.text.apply(str)
        data.columns = ["text", "property"]
        return data

    def get_text_to_transform(self):
        return self.load_data(self.target_language, self.prop, "test")

    def train_data(self):
        source_train = self.load_data(self.source_language, self.prop, "train")
        target_train = self.load_data(self.target_language, self.prop, "train")
        source_test = self.load_data(self.source_language, self.prop, "test")

        return source_train, target_train, source_test


class TrustPilot(Dataset):

    def __init__(self, source_language, target_language, prop, folder_path, **kwargs):
        super().__init__(source_language, target_language, **kwargs)

        self.prop = prop
        self.base_folder = folder_path

    def load_data(self, language, prop, task):
        data = pd.read_csv(f"{self.base_folder}/{task}/{language}.csv")

        data = data[["text", prop]]
        data["text"] = data.text.apply(str)
        data.columns = ["text", "property"]
        return data

    def get_text_to_transform(self):
        return self.load_data(self.target_language, self.prop, "test")

    def train_data(self):
        source_train = self.load_data(self.source_language, self.prop, "train")
        target_train = self.load_data(self.target_language, self.prop, "train")
        source_test = self.load_data(self.source_language, self.prop, "test")

        return source_train, target_train, source_test


class Affect(Dataset):

    def __init__(self, source_language, target_language, folder_path, task="emotion", **kwargs):
        super().__init__(source_language, target_language, **kwargs)

        self.folder_path = folder_path
        self.text_to_translate = None
        self.task = task

    def clean_text(self, text):
        text = re.sub(r"\B@\w+", "@user", text)
        return text

    def load_data(self, language, task):
        data = pd.read_csv(f"{self.folder_path}/{task}/{language}.csv")
        data = data[["text", "emotion"]]
        if self.task == "sentiment":
            data["emotion"] = data["emotion"].replace({"joy": "positive",
                                                       "anger": "negative",
                                                       "sadness": "negative",
                                                       "fear": "negative"})

        data["text"] = data.text.apply(str)

        data["text"] = data.text.apply(self.clean_text)

        data.columns = ["text", "property"]
        return data


    def get_text_to_transform(self):
        if self.text_to_translate is None:
            self.text_to_translate = self.load_data(self.source_language, "test")
            return self.text_to_translate
        else:
            return self.text_to_translate

    def train_data(self):
        source_train = self.load_data(self.source_language, "train")
        target_train = self.load_data(self.target_language,  "train")
        source_test = self.get_text_to_transform()
        return source_train, target_train, source_test


class HatEval(Dataset):

    def __init__(self, source_language, target_language, folder_path, **kwargs):
        super().__init__(source_language, target_language, **kwargs)

        self.folder_path = folder_path
        self.text_to_translate = None

    def clean_text(self, text):
        text = re.sub(r"\B@\w+", "@user", text)
        text = re.sub("#[A-Za-z0-9_]+", "", text)
        return text

    def load_data(self, language, task):
        data = pd.read_csv(f"{self.folder_path}/{task}/{language}.csv")
        data = data[["text", "HS"]]

        data["HS"] = data["HS"].replace({1: "Hate", 0: "Not Hate"})

        data["text"] = data.text.apply(str)

        data["text"] = data.text.apply(self.clean_text)
        data.columns = ["text", "property"]
        return data


class HateEvalPara(Dataset):

    def __init__(self, source_language, target_language, folder_path, **kwargs):
        super().__init__(source_language, target_language, common_classifier=True, **kwargs)

        self.folder_path = folder_path
        self.text_to_translate = None

    def clean_text(self, text):
        text = re.sub(r"\B@\w+", "@user", text)
        text = re.sub("#[A-Za-z0-9_]+", "", text)
        return text

    def load_data(self, language, task):
        data = pd.read_csv(f"{self.folder_path}/{task}/{language}.csv")
        data = data[["text", "HS"]]

        data["HS"] = data["HS"].replace({1: "Hate", 0: "Not Hate"})

        data["text"] = data.text.apply(str)

        data["text"] = data.text.apply(self.clean_text)

        data.columns = ["text", "property"]
        return data

    def get_text_to_transform(self):
        if self.text_to_translate is None:
            self.text_to_translate = self.load_data(self.source_language, "test")
            return self.text_to_translate
        else:
            return self.text_to_translate

    def train_data(self):
        source_train = self.load_data(self.source_language, "train")
        target_train = self.load_data(self.target_language,  "train")
        source_test = self.get_text_to_transform()
        return source_train, target_train, source_test
