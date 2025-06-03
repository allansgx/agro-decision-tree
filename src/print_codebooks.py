import joblib

# Carregar o pacote salvo
bundle = joblib.load("models/tree.pkl")

encoders  = bundle["encoders"]
target_le = bundle["target_le"]

def print_codebooks(encoders, target_le):
    print("=== Alvo (cultivo) ===")
    for indice, nome_item in enumerate(target_le.classes_):
        print(f"{indice}: {nome_item}")
    print("\n=== Features ===")
    for nome_coluna, encoder in encoders.items():
        codigos = ", ".join(f"{i}:{label}" for i, label in enumerate(encoder.classes_))
        print(f"{nome_coluna}: {codigos}")

print_codebooks(encoders, target_le)
