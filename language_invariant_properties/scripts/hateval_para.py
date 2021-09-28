from language_invariant_properties.lip import HateEvalPara
import pandas as pd

tp = HateEvalPara("english", "english",
                  folder_path="/home/vinid/PycharmProjects/language_invariant_properties/language_invariant_properties/data/semeval_para",
                  sentence_embedding=True, common_classifier=True)

k = tp.get_text_to_transform()["text"].values

t = pd.read_csv("/home/vinid/PycharmProjects/language_invariant_properties/language_invariant_properties/data/semeval_para/transformed/english.csv")["text"].values.tolist()

print(tp.score(t))
tp.plot("/home/vinid/PycharmProjects/language_invariant_properties/plots/hate.pdf")

