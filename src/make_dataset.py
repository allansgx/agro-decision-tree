import pandas as pd, random, pathlib, argparse

def generate(n=100, path="data/raw/dataset.csv"):
    solos = ['arenoso','argiloso','misto']
    temps = ['baixa','média','alta']
    chuvas = ['seca','moderada','úmida']
    estacoes = ['verão','outono','inverno','primavera']
    regioes = ['norte','nordeste','sul','sudeste','centro-oeste']
    demandas = ['baixa','média','alta']
    exp = ['iniciante','intermediário','experiente']
    culturas = ['milho','feijão','mandioca','soja','arroz']

    rows = []
    for _ in range(n):
        s,t,c,e,r,d,x = map(random.choice,
            [solos,temps,chuvas,estacoes,regioes,demandas,exp])
        # regra simples
        if r=='sul' and c=='úmida': cult='feijão'
        elif s=='arenoso' and t=='alta': cult='milho'
        elif d=='alta' and e=='verão': cult='soja'
        elif x=='iniciante' and d=='baixa': cult='mandioca'
        else: cult=random.choice(culturas)
        rows.append([s,t,c,e,r,d,x,cult])

    df = pd.DataFrame(rows, columns=[
        'tipo_solo','temperatura','chuva','estacao','regiao',
        'demanda_mercado','experiencia_agricultor','cultura_recomendada'])
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,index=False)
    print(f"✔ Dataset salvo em {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--rows",type=int,default=100)
    parser.add_argument("-o","--output",default="data/raw/dataset.csv")
    args = parser.parse_args()
    generate(args.rows,args.output)
