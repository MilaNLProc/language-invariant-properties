import pandas as pd
from language_invariant_properties.abstractions.dataset import Dataset
import os

class TrustPilot(Dataset):

    def __init__(self, source_language, target_language, prop):
        super().__init__(source_language, target_language)

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


class SemEval(Dataset):

    def __init__(self, source_language, target_language, folder_path):
        super().__init__(source_language, target_language)

        self.folder_path = folder_path
        self.text_to_translate = None

    def load_data(self, language, task):
        data = pd.read_csv(f"{self.folder_path}/{task}/{language}.csv")
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



