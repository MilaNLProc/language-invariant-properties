from language_invariant_properties.lip import Affect
import pandas as pd

tp = Affect("spanish", "english",

             "/home/vinid/PycharmProjects/language_invariant_properties/language_invariant_properties/data/affect",
    task="sentiment",
             cross_lingual_st=True)

k = (tp.get_text_to_translate()["text"].values.tolist())

t = pd.read_csv("/home/vinid/PycharmProjects/language_invariant_properties/language_invariant_properties/data/affect/translated/english.csv")
print(tp.score(t["text"].values))
tp.plot("/home/vinid/PycharmProjects/language_invariant_properties/plots/hate.pdf")
