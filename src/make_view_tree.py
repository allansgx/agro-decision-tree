import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib

model_bundle = joblib.load("models/tree.pkl")

# Extrai a árvore de decisão treinada
clf = model_bundle["model"]
# Extrai o dicionário com os LabelEncoders, que foram usados para transformar texto (como "arenoso") em números.
encoders = model_bundle["encoders"]
# Converte os números das folhas da árvore de volta para nomes das culturas (milho, soja etc).
target_le = model_bundle["target_le"]

fig = plt.figure(figsize=(40,20))
plot_tree(
    clf,
    feature_names=list(encoders.keys()),
    class_names=target_le.classes_,
    filled=True,
    rounded=True
)
fig.savefig("models/tree_highres.png", dpi=300, bbox_inches="tight")
# fig.savefig("models/tree.svg", bbox_inches="tight")   # vetor, zoom infinito