import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib

model_bundle = joblib.load("models/tree.pkl")
clf = model_bundle["model"] # a árvore
encoders = model_bundle["encoders"] # dict col→LabelEncoder
target_le = model_bundle["target_le"] # encoder do alvo

fig = plt.figure(figsize=(40,20))   # aumente o canvas
plot_tree(
    clf,
    feature_names=list(encoders.keys()),
    class_names=target_le.classes_,
    filled=True, rounded=True
)
fig.savefig("models/tree_highres.png", dpi=300, bbox_inches="tight")
fig.savefig("models/tree.svg", bbox_inches="tight")   # vetor, zoom infinito