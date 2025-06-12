import streamlit as st
import joblib
import pandas as pd

bundle = joblib.load("models/tree.pkl")
clf, encoders, le_y = bundle["model"], bundle["encoders"], bundle["target_le"]

st.title("🌱 Recomendação de Cultivo")

# Coleta de inputs do usuário
inputs = {}
for coluna, encoder in encoders.items():
    opcoes = list(encoder.classes_)
    pergunta = coluna.replace("_", " ").title()
    inputs[coluna] = st.selectbox(pergunta, opcoes)

# Botão para gerar recomendação
if st.button("Recomendar"):
    # Cria um DataFrame com os inputs
    df_input = pd.DataFrame([inputs])
    
    # Aplica os encoders para transformar os dados
    for col in df_input.columns:
        df_input[col] = encoders[col].transform(df_input[col])
    
    # Faz a previsão com o modelo
    predicao = clf.predict(df_input)
    cultivo_recomendado = le_y.inverse_transform(predicao)[0]

    # Exibe o resultado
    st.success(f"✅ Cultivo recomendado: **{cultivo_recomendado.upper()}**")