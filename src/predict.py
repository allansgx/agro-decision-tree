import joblib, pandas as pd, json, sys, pathlib

arg = sys.argv[1]
if pathlib.Path(arg).suffix == ".json":
    with open(arg, "r", encoding="utf-8") as f:
        sample = json.load(f)
else:
    sample = json.loads(arg)


model_bundle = joblib.load("models/tree.pkl")
clf = model_bundle["model"] # a árvore
encoders = model_bundle["encoders"] # dict col→LabelEncoder
target = model_bundle["target_le"] # encoder do alvo

# dados via JSON na linha de comando
df = pd.DataFrame([sample])

for col in df.columns:
    df[col] = encoders[col].transform(df[col])

pred = target.inverse_transform(clf.predict(df))[0]
print("Recomendação:", pred)
