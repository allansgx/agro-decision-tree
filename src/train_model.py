import pandas as pd, joblib, pathlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    dataset = pd.read_csv("data/raw/dataset.csv")

    features = dataset.drop("cultivo_recomendado", axis=1)
    target = dataset["cultivo_recomendado"]

    # Codificar variáveis categóricas das features
    feature_encoders = {}
    for column in features.columns:
        encoder = LabelEncoder()
        features[column] = encoder.fit_transform(features[column])
        feature_encoders[column] = encoder

    # Codificar variável alvo (target)
    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(target)

    # Separar dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Criar e treinar o modelo de árvore de decisão
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train,y_train)

    accuracy = decision_tree.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    pathlib.Path("models").mkdir(exist_ok=True)
    joblib.dump({
        "model": decision_tree,
        "encoders": feature_encoders,
        "target_le": target_encoder
    }, "models/tree.pkl")

    print("✔ Modelo salvo em models/tree.pkl")

if __name__ == "__main__":
    main()
