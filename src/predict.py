import joblib, pandas as pd, json, sys, pathlib

arg = sys.argv[1]
if pathlib.Path(arg).suffix == ".json":
    with open(arg, "r", encoding="utf-8") as f:
        sample = json.load(f)
else:
    sample = json.loads(arg)


bundle = joblib.load("models/tree.pkl")
clf = bundle["model"]
encoders = bundle["encoders"]
target = bundle["target_le"]

df = pd.DataFrame([sample])

for col in df.columns:
    df[col] = encoders[col].transform(df[col])

pred = target.inverse_transform(clf.predict(df))[0]
print("Recomendação:", pred)
