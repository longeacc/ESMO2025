from gliner import GLiNER

model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")

text = "Évolutions technologiques en qualification biologique du don et leur impact sur le risque résiduel transfusionnel ."

labels = ["PROC", "date", "organization", "location"]

entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(entity["text"], "=>", entity["label"])


