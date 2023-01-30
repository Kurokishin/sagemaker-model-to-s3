import json
import boto3
import numpy as np
import sagemaker
from sagemaker.serializers import CSVSerializer

def predicao():
    # Recebe as acess_keys da AWS do usuário
    with open("credentials.json", "r") as file:
        keys = json.load(file)

    # Inicializa a integração com o AWS SageMaker
    boto_session = boto3.Session(
                    aws_access_key_id= keys["aws_access_key_id"],
                    aws_secret_access_key= keys["aws_secret_access_key"],
                    region_name="us-east-1")

    sm_session = sagemaker.Session(boto_session=boto_session)

    # Busca o endpoint na AWS para acionar a predição
    previsor = sagemaker.predictor.Predictor(endpoint_name='xgboost-2023-01-27-14-33-35-018', sagemaker_session=sm_session)
    previsor.serializer = CSVSerializer()

    # Array de teste, substituir pelo JSON convertido solicitado na avaliação
    arrayteste = np.array([1, 0, 0, 2017, 10, 19, 0])

    # Predição propriamente dita
    previsao = float(previsor.predict(arrayteste).decode('utf-8'))

    return previsao
