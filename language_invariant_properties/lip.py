import pandas as pd
from language_invariant_properties.abstractions.dataset import Dataset
import os
import re

class TrustPilotPara(Dataset):

    def __init__(self, source_language, target_language, prop, **kwargs):
        dataset_name = "trustpilot_" + prop
        super().__init__(source_language, target_language, dataset_name, common_classifier=True, **kwargs)

        self.prop = prop
        self.base_folder = "trustpilot"

    def load_data(self, language, prop, task):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        data = pd.read_csv(f"{root_dir}/data/{self.base_folder}/{task}/{language}.csv")

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


class TrustPilot(Dataset):

    def __init__(self, source_language, target_language, prop, **kwargs):
        dataset_name = "trustpilot_" + prop
        super().__init__(source_language, target_language, dataset_name, **kwargs)

        self.prop = prop
        self.base_folder = "trustpilot"

    def load_data(self, language, prop, task):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        data = pd.read_csv(f"{root_dir}/data/{self.base_folder}/{task}/{language}.csv")

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


class Affect(Dataset):

    def __init__(self, source_language, target_language, folder_path, task="emotion", **kwargs):
        dataset_name = "semeval19t5"
        super().__init__(source_language, target_language, dataset_name, **kwargs)

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

        #g = data.groupby('HS')
        #data = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

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


class HatEval(Dataset):

    def __init__(self, source_language, target_language, folder_path, **kwargs):
        dataset_name = "semeval19t5"
        super().__init__(source_language, target_language, dataset_name, **kwargs)

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


class SemEvalPara(Dataset):

    def __init__(self, source_language, target_language, folder_path, **kwargs):
        dataset_name = "semeval19t5"
        super().__init__(source_language, target_language, dataset_name, common_classifier=True, **kwargs)

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
