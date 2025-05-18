import pandas as pd, joblib, pathlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    df = pd.read_csv("data/raw/dataset.csv")
    X = df.drop("cultura_recomendada", axis=1)
    y = df["cultura_recomendada"]

    encoders = {}
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    Xtr,Xte,Ytr,Yte = train_test_split(X,y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(Xtr,Ytr)
    acc = clf.score(Xte,Yte)
    print(f"Accuracy: {acc:.2f}")

    pathlib.Path("models").mkdir(exist_ok=True)
    joblib.dump({"model":clf,"encoders":encoders,"target_le":le_y},
                "models/tree.pkl")
    print("✔ Modelo salvo em models/tree.pkl")

if __name__ == "__main__":
    main()
