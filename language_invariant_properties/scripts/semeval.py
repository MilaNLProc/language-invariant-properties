from tqdm import tqdm
from language_invariant_properties.lip import SemEval
from transformers import MarianTokenizer, MarianMTModel
from transformers import pipeline

tp = SemEval("spanish", "english", "/home/vinid/PycharmProjects/language_invariant_properties/language_invariant_properties/data/semeval")

k = (tp.get_text_to_translate()["text"].values.tolist())

mname = 'Helsinki-NLP/opus-mt-es-en'

model = MarianMTModel.from_pretrained(mname)
tok = MarianTokenizer.from_pretrained(mname)
translation = pipeline("translation_es_to_en", model=model, tokenizer=tok)
t = []

def batch(iterable, n = 1):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


pbar = tqdm(total=len(k)//10)

for n in batch(k, 10):
    t.extend([a["translation_text"] for a in translation(n)])
    pbar.update(1)

print(tp.score(t))
