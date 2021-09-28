from language_invariant_properties.lip import TrustPilotPara
import pandas as pd

tp = TrustPilotPara("english", "english", "gender",
                    sentence_embedding=True)
k = tp.get_text_to_transform()["text"].values

t = pd.read_csv("/home/vinid/PycharmProjects/language_invariant_properties/language_invariant_properties/data/trustpilot_para/transformed/english.csv")["text"].values.tolist()

print(tp.score(t))
tp.plot("/home/vinid/PycharmProjects/language_invariant_properties/plots/trustpara.pdf")

