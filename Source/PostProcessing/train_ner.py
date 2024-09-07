from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("PaDaS-Lab/gbert-legal-ner")
model = AutoModelForTokenClassification.from_pretrained("PaDaS-Lab/gbert-legal-ner")

ner = pipeline("ner", model=model, tokenizer=tokenizer)


example = "(3) Ergänzend zu § 543 Abs. 2 Satz 1 Nr. 3 gilt: 1.Im Falle des § 543 Abs. 2 Satz 1 Nr. 3 Buchstabe a ist der rückständige Teil der Miete nur dann als nicht unerheblich anzusehen"

results = ner(example)

for i in results:
    print(i)

