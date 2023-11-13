from flask import Flask, request
import flask
from joblib import load
import pandas as pd
import numpy as np

# Iniciando a aplcação.
app = Flask(__name__)

def database(files_directory:str='project/data/')->pd.DataFrame:
    '''
        Método que trará toda a base de clientes para previsão do algoritmo.

        Parâmetro
        ---------
        files_directory: str
            Diretório onde os arquivos se encontram.

        Retorna
        -------
        Um `pd.DataFrame` com toda a base do projeto para previsão.
    '''
    train = pd.read_parquet(files_directory+'train.parquet').drop('TARGET', axis=1) # Removendo TARGET.
    test = pd.read_parquet(files_directory+'test.parquet')
    return pd.concat((train, test), axis=0)

@app.route('/', methods=['POST', 'GET'])
def home():
    '''
        Função que disponibiliza a página inicial da API. 
    '''
    return flask.render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    pass

if __name__ == '__main__':
    #print(database().head())