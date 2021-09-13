from language_invariant_properties.lip import TrustPilot
import pandas as pd

tp = TrustPilot("italian", "english", "gender")
k = tp.get_text_to_translate()["text"].values

t = pd.read_excel("https://github.com/MilaNLProc/translation_bias/raw/master/data/it/it_TEST.xlsx")["google_translation"].values.tolist()

print(tp.score(t))
tp.plot()
