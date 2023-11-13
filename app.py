from flask import Flask, request
import flask
from joblib import load
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

# Iniciando a aplcação.
app = Flask(__name__)

class ModelPredict:
    def __init__(self, files_directory:str='project/data/', model_path:str='project/models/PIPELINE.joblib'):
        '''
            Classe com a qual invocaremos o modelo do projeto para fazermos as previsões.

            Função
            -------
            `model_predict`: Aciona o modelo e salva os dados da previsão em uma base .parquet.
        '''
        self.files_directory = files_directory
        self.model_path = model_path

    def _databases(self)->Tuple[pd.DataFrame]:
        '''
            Método que trará toda a base de clientes para previsão do algoritmo e histórico de estimativas.
        '''
        # Base de clientes.
        train = pd.read_parquet(self.files_directory+'train.parquet').drop('TARGET', axis=1) # Removendo TARGET.
        test = pd.read_parquet(self.files_directory+'test.parquet')
        df_clients = pd.concat((train, test), axis=0)

        # Base de histórico das previsões.
        df_hist = pd.read_parquet(self.files_directory+'prediction_logs.parquet')
        return df_clients, df_hist
    
    def model_predict(self, sk_id_curr:int)->float:
        '''
            Método que acionará o modelo do projeto, fazendo-o realizar as previsões com base no ID do cliente.

            Parâmetro
            ---------
            sk_id_curr: int
                ID do cliente.

            Retorna
            -------
            A probabilidade de o cliente pagar o empréstimo.
        '''
        df_clients, df_hist = self._databases() # Carregando os dados de clientes e histórico.

        # Carregando o modelo e obtendo a probabilidade de o cliente pagar o empréstimo.
        datapoint = df_clients[df_clients['SK_ID_CURR']==sk_id_curr].to_numpy().reshape(1, -1)
        
        # Remover `pipe` e `proba` em Pipeline depois.
        proba = load(self.model_path).predict_proba(datapoint)[0][0]

        # Cadastrando os dados da previsão em um .parquet próprio.
        df_hist.loc[len(df_hist)] = [sk_id_curr, proba, datetime.now()]
        df_hist.to_parquet(self.files_directory+'prediction_logs.parquet')
        return proba

@app.route('/', methods=['POST', 'GET'])
def home():
    '''
        Função que disponibiliza a página inicial da API. 
    '''
    return flask.render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
        Função que disponibiliza a página com a previsão do modelo.
    '''
    # Extraindo o ID do cliente informado pelo usuário.
    sk_id_curr = int(request.form.to_dict()['SK_ID_CURR']) 

    # Estimando a probabilidade de pagamento para o cliente.
    proba = ModelPredict().model_predict(sk_id_curr)
    return flask.render_template('predict.html', proba=f'{proba:.2%}')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=2023, debug=False)