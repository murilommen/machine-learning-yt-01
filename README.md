# Treinando um modelo de Machine Learning

Esse repositório foi criado para ter um exemplo super simples, como um "hello world" para modelos de Machine Learning. 

## Preparação do ambiente (MacOS/Linux)
Garanta que você tenha na sua máquina o `python>=3.7` e o gerenciador de pacotes `pip`, e então clone este repositório, mude para o diretório dele, e execute:

```bash
$ python3 -m venv .env
$ source .env/bin/activate
$ pip3 install -r requirements.txt
```

OBS.: Para usuários de Windows, o processo é semelhante, mudando apenas o comando para ativar o ambiente virtual, com: `.env/Scripts/activate.bat`

## Executando o projeto
O projeto foi dividido em três módulos Python (`train.py`, `evaluate.py`, `predict.py`) e um auxiliar para reutilizar alguns métodos.

1) Evaluate

Comece executando esse script, da seguinte forma:
```bash
$ python3 evaluate.py
```
E vá interagindo com o argumento `max_depth` no arquivo `evaluate.py` e veja como isso afeta a performance do modelo com os prints das métricas selecionadas no seu terminal.

2) Train
Execute esse script, alterando o valor de `max_depth` para o desejado e executando:

```bash
$ python3 train.py
```
Você perceberá que um novo arquivo `tree_classifier.pkl` surgiu no seu diretório raiz. É o objeto do modelo serializado, que será reutilizado no momento das predições.

3) Predict
Finalmente, esse módulo irá gerar 10 linhas aleatórias em uma *array* e fará as predições em cima dela, retornando uma lista com o resultado das predições no seu terminal.

## Sugestões de próximos passos como estudos
- [ ] Plote a importância das variáveis para sua análise
- [ ] Faça alguma transformação nas features e veja como a performance é afetada
- [ ] Realize uma validação cruzada para verificar se o seu modelo está superestimado
- [ ] Pegue um projeto mais desafiador na plataforma Kaggle, e estenda esse trabalho
