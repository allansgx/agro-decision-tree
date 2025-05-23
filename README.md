# agro-decision-tree

## Passo a passo para entrar/sair do ambiente virtual
- `Criar pastas do ambiente:` python -m venv .venv
- `Entrar no ambiente:` .\.venv\Scripts\Activate.ps1
- `Instalar dependências:` pip install --upgrade pip `OU` python.exe -m pip install --upgrade pip
- `Sair do ambiente:` deactivate

## Libs necessárias (baixar dentro do ambiente):
- pip install pandas scikit-learn matplotlib jupyterlab
- `(Se fizer interface):`pip install streamlit gradio

## Informações
Sempre que instalar algo novo no venv (ex.: streamlit), rode novamente `pip freeze > requirements.txt` para manter o arquivo atualizado.
Assim, qualquer pessoa poderá recriar o ambiente com:
- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`
- `pip install -r requirements.txt`

## --- FLUXO COMPLETO
- `Rodar notebook:` jupyter lab notebooks/01_exploracao.ipynb
- `Gerar ou atualizar dados:` python src/make_dataset.py -n 150
- `Treinar modelo:` python src/train_model.py
- `Predição via CLI:` python src\predict.py amostra.json
- `Rodar Interface Streamlit:` streamlit run app/streamlit_app.py

## Outros comandos:
- `Gerar imagem da árvore de decisão:` .\src\view_tree.py
- `Visualizar códigos do encoder:` .\src\print_codebooks.py
