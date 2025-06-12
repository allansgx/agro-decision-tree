import pandas, random, pathlib, argparse

def generate(n=100, path="data/raw/dataset.csv"):
    # Opções disponíveis para gerar conjuntos
    solos = ['arenoso','argiloso','misto']
    temps = ['baixa','média','alta']
    chuvas = ['seca','moderada','úmida']
    estacoes = ['verão','outono','inverno','primavera']
    regioes = ['norte','nordeste','sul','sudeste','centro-oeste']
    demandas = ['baixa','média','alta']
    exp = ['iniciante','intermediário','experiente']
    cultivos = ['milho','feijão','mandioca','soja','arroz']

    rows = []
    for _ in range(n):
        # Pegar um valor aleatório de cada array
        solo,temp,chuva,estacao,regiao,demanda,experiencia = map(random.choice,
            [solos,temps,chuvas,estacoes,regioes,demandas,exp])
        
        # Regra para definir cultivo recomendado
        if regiao == 'sul' and chuva == 'úmida' and temp != 'alta':
            cultivo = 'feijão'
        elif solo == 'arenoso' and temp == 'alta' and demanda != 'baixa':
            cultivo = 'milho'
        elif demanda == 'alta' and estacao == 'verão' and experiencia != 'iniciante':
            cultivo = 'soja'
        elif experiencia == 'iniciante' and demanda == 'baixa' and regiao in ['norte', 'nordeste']:
            cultivo = 'mandioca'
        elif solo == 'argiloso' and chuva == 'moderada':
            cultivo = 'arroz'
        elif solo == 'misto' and regiao == 'centro-oeste':
            cultivo = 'soja'
        else:
            cultivo = random.choice(cultivos)

        # Adicionar o conjunto aleatório e o cultivo recomendado em um array
        rows.append([solo,temp,chuva,estacao,regiao,demanda,experiencia,cultivo])

    # Criar o dataset em csv
    df = pandas.DataFrame(rows, columns=[
        'tipo_solo','temperatura','chuva','estacao','regiao',
        'demanda_mercado','experiencia_agricultor','cultivo_recomendado'])
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,index=False)
    print(f"✔ Dataset salvo em {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--rows",type=int,default=100)
    parser.add_argument("-o","--output",default="data/raw/dataset.csv")
    args = parser.parse_args()
    generate(args.rows,args.output)
