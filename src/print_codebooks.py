import joblib

# Carregar o pacote salvo
bundle = joblib.load("models/tree.pkl")

encoders  = bundle["encoders"]     # dict de features
target_le = bundle["target_le"]    # encoder do alvo

def print_codebooks(encoders, target_le):
    print("=== Alvo (cultura) ===")
    for i, cls in enumerate(target_le.classes_):
        print(f"{i}: {cls}")
    print("\n=== Features ===")
    for col, le in encoders.items():
        codes = ", ".join(f"{i}:{label}" for i, label in enumerate(le.classes_))
        print(f"{col}: {codes}")

print_codebooks(encoders, target_le)
