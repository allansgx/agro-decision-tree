import streamlit as st, joblib, pandas as pd

bundle = joblib.load("models/tree.pkl")
clf, encoders, le_y = bundle["model"], bundle["encoders"], bundle["target_le"]

st.title("Recomendação de Cultivo")
inputs = {}
for col, le in encoders.items():
    opts = list(le.classes_)
    inputs[col] = st.selectbox(col.replace("_"," ").title(), opts)

if st.button("Recomendar"):
    df = pd.DataFrame([inputs])
    for c in df.columns:
        df[c] = encoders[c].transform(df[c])
    pred = le_y.inverse_transform(clf.predict(df))[0]
    st.success(f"Cultura recomendada: **{pred.upper()}**")
