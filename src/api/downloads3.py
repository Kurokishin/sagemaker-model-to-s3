import json
import boto3

def downloads3():
    # Recebe as acess_keys da AWS do usuário
    with open("credentials.json", "r") as file:
        keys = json.load(file)

    # Baixa o modelo do s3
    s3 = boto3.client('s3',
                    aws_access_key_id= keys["aws_access_key_id"],
                    aws_secret_access_key= keys["aws_secret_access_key"],
                    region_name="us-east-1")
    s3.download_file('modelo-treinado-grupo4', 'modelos/model.h5', 'model.h5')