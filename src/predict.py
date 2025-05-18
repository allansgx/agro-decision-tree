import joblib, pandas as pd, json, sys, pathlib

arg = sys.argv[1]
if pathlib.Path(arg).suffix == ".json":
    with open(arg, "r", encoding="utf-8") as f:
        sample = json.load(f)
else:
    sample = json.loads(arg)


model_bundle = joblib.load("models/tree.pkl")
clf = model_bundle["model"]
encoders = model_bundle["encoders"]
le_y = model_bundle["target_le"]

# dados via JSON na linha de comando
df = pd.DataFrame([sample])

for col in df.columns:
    df[col] = encoders[col].transform(df[col])

pred = le_y.inverse_transform(clf.predict(df))[0]
print("Recomendação:", pred)
