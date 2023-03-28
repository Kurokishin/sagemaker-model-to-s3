# Avaliação Sprint 5 - Programa de Bolsas Compass UOL / AWS e IFCE

[![N|Solid](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/LogoCompasso-positivo.png/440px-LogoCompasso-positivo.png)](https://compass.uol/pt/home/)

Avaliação da quinta sprint do programa de bolsas Compass UOL para formação em machine learning para AWS.

---

## Sumário
* [Objetivo](#objetivo)
* [Ferramentas](#ferramentas)
* [Executar a aplicação localmente](#executar-localmente-a-aplicação)
* [Desenvolvimento](#desenvolvimento)
    * [Tratamento do dataframe](#tratamento-do-dataframe)
    * [Concepção do modelo](#concepção-do-modelo)
    * [API](#api)
    * [Build da aplicação](#build-da-aplicação)
    * [Elastic Beanstalk](#deploy-no-elastic-beanstalk)
* [Dificuldades](#dificuldades)
* [Autores](#autores)

---

## Objetivo

Utilizar o dataset [Hotel Reservations](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) para a classificação das reservas dos clientes em determinadas faixas de preço.

---

## Ferramentas

* [Hotel Reservations](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) base de dados referente a reservas em hotéis.
* [AWS](https://aws.amazon.com/pt/) plataforma de computação em nuvem da Amazon.
    * [SageMaker](https://aws.amazon.com/sagemaker/) plataforma que fornece ferramentas para construção, treinamento e *deploy* de modelos de aprendizado de máquina.
    * [S3](https://aws.amazon.com/s3/) serviço de armazenamento.
* [Docker](https://www.docker.com/) plataforma de virtualização de software em contêineres.

---

## Executar localmente a aplicação
No mesmo diretório onde está os arquivos docker execute:

```
docker compose up
```

---

## Desenvolvimento

### Tratamento do dataframe

Primeiramente é realizado a leitura dos dados do dataframe por meio do **pandas**.

```py
df = pd.read_csv("dataframe/Hotel Reservations.csv")
```

Então é criado um heatmap para buscar correlações entre o preço médio por quarto e as demais colunas.

```py
figura = plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True);
```

Por meio do gráfico gerado é possível filtrar as colunas que não ajudarão no processo de treinamento do modelo e removê-las.

```py
df.drop(columns = ['Booking_ID', 'booking_status', 'repeated_guest','type_of_meal_plan',  'arrival_date', 'market_segment_type', 'lead_time', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled'], axis = 1, inplace = True)
```

Algumas dessas colunas possuíam informações que serviram para alimentar o modelo mas que não estavam em formato numérico, para tal foi utilizado o método de dummies.

```py
colunas = ['room_type_reserved', 'arrival_year', 'arrival_month']
df = pd.get_dummies(df, prefix = colunas, columns = colunas)
```

Seguindo os requisitos necessários para a avaliação,foi feita a criação da coluna **label_avg_price_per_room** e a sua relação com a coluna **avg_price_per_room**, assim como a definição das condições que influenciam no resultado final do modelo.

```py
df['label_avg_price_per_room'] = df['avg_price_per_room'].apply(avg_set_labels)
df['label_avg_price_per_room']

def avg_set_labels(avg):
    if avg <= 85:
        return 0
    elif 85 < avg < 115:
        return 1
    else:
        return 2
```

Por fim o tratamento é salvo em um arquivo chamado **Hotel Reservations tratado.csv**.

```py
df.to_csv("dataframe/Hotel Reservations tratado.csv", index=False)
```

### Concepção do modelo

Separação dos dados de treinamento e teste.

```py
from sklearn.model_selection import train_test_split
# Contém os dados de todas as colunas para treinamento com exceção da label_avg_price_per_room
X = df.iloc[:,1:len(df.columns)]

# Variável alvo para conter os valores da coluna label_avg_price_per_room
y = df.iloc[:,0]
```

A criação do modelo de rede neural se dá pelos seguintes passos:

1. Adição de camadas densas conectadas que são ativadas pela função ReLu.
2.  Duas camadas de *dropout* são adicionadas para prevenir o problema de *overfittng*. A técnica consiste em desligar uma proporção específica de neurônios na camada durante o treinamento.
3. Uma camada final com 3 neurônios é criada e utiliza a função de ativação **softmax**, comumente utilizada para problemas de classificação de multiclasses.

```py
model = keras.Sequential()
model.add(layers.Dense(54, input_shape=(X.shape[1],), activation='relu'))
model.add(layers.Dense(27, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(27, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(27, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
```
O modelo é então compilado e utiliza da acurácia como métrica para medir a eficácia do mesmo.

```py
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
```

Após as definições de criação do modelo ocorre o treinamento em si

```py
sequential = model.fit(X_treino, y_treino, epochs=50, batch_size=256, validation_data=(X_teste, y_teste))
```
> **X_treino** e **y_treino** são variáveis de treinamento e alvo, respectivamente.

> **epochs** são a quantidade de iterações nos dados de treinamento.

> **batch_size** refere-se a quantidade de amostras em uma iteração, ou seja, a atualização dos pesos. Quanto menor o tamanho do lote o modelo possuirá maior acurácia, mas o treinamento será mais lento, enquanto que com uma quantidade maior o treinamento será menos acertivo, contudo mais rápido.

> **validation_data** contém os dados de validação e os alvos. Assim o modelo é monitorado durante seu treinamento para evitar o *overfitting*.

> O método **model.fit** traz as informações sobre todo o processo de treino.

A acurácia obtida no fim do processo foi de aproximadamente 64%.

O modelo treinado foi salvo no arquivo **model.h5** e enviado para um bucket na AWS S3 por meio da biblioteca **boto3**

```py
model.save('model.h5')

s3 = boto3.client('s3')
with open('model.h5', 'rb') as file:
    s3.upload_fileobj(file, 'modelo-treinado-grupo4', 'modelos/model.h5')
```

### API

Conforme requisitado na avaliação, uma API foi criada utilizando o framework [Flask](https://flask.palletsprojects.com/en/2.2.x/) para expor um *endpoint* que mostra o resultado da inferência do modelo diante dos dados do arquivo de [input](https://github.com/Compass-pb-aws-2022-IFCE/sprint-5-pb-aws-ifce/blob/Grupo-4/src/api/input.json).

No [main](https://github.com/Compass-pb-aws-2022-IFCE/sprint-5-pb-aws-ifce/blob/Grupo-4/src/api/main.py) um servidor foi criado ouvindo a porta 8080 e a rota **/api/v1/predict** escolhida para mostrar o resultado.

```py
app = Flask(__name__)

@app.route('/api/v1/predict', methods=["POST"])

# Definindo parâmetros de retorno e tratamento de erro
def predict():
    try:
            predict = predicao()
            resposta = {"result": int(predict)}
           
            return jsonify(resposta)

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True)
```
A autorização de acesso ao bucket é feito por chaves secretas do criador do mesmo. Esse processo pode ser visto em [downloads](https://github.com/Compass-pb-aws-2022-IFCE/sprint-5-pb-aws-ifce/blob/Grupo-4/src/api/downloads3.py).

```py
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
```
O [endpoint](https://github.com/Compass-pb-aws-2022-IFCE/sprint-5-pb-aws-ifce/blob/Grupo-4/src/api/endpoint.py) recebe o JSON com os dados a serem utilizados na predição. Estes são carregados juntamente com o modelo e enviados para o servidor.

```py
def predicao():

    downloads3()
    
    # Recebe o json com os dados a serem enviados para a predição 
    input_data = json.load(open('input.json'))

    model = tf.keras.models.load_model('model.h5')

    previsao = model.predict(np.array([list(input_data.values())]))
    
    previsao = list(previsao[0]).index(max(previsao[0]))

    return previsao + 1
```
### Build da aplicação

O [arquivo docker](https://github.com/Compass-pb-aws-2022-IFCE/sprint-5-pb-aws-ifce/blob/Grupo-4/src/api/Dockerfile) possui todas as instruções necessárias para a contrução da imagem.

```Dockerfile
FROM python:3.10
ADD . /app
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8080
ENV FLASK_APP=main.py
CMD ["flask", "run", "--host=0.0.0.0"]
```

### Deploy no Elastic Beanstalk

Segue o lançamento da aplicação de maneira online na plataforma da AWS:

1. Crie a aplicação.
![criando aplicação](https://user-images.githubusercontent.com/80788425/217838482-794e5902-1600-4fbf-a20a-31f52e774d45.png)

2. Configure a plataforma e comprima os arquivos referentes a API e o modelo.
![configurando a aplicação](https://user-images.githubusercontent.com/80788425/217839256-68fe2ccd-6764-4490-85b6-5456158e8daf.png)

3. Após alguns minutos de configurações que a plataforma da AWS faz o resultado é o seguinte:

![resultado](https://user-images.githubusercontent.com/80788425/217840308-318a7541-66c4-474a-b7c4-ebee624ccc77.jpeg)

A aplicação está disponível [aqui](http://avsprint5-env.eba-widjye74.us-east-1.elasticbeanstalk.com/).

---

## Dificuldades

* Acesso ao bucket por parte dos membros da equipe.
* Deploy da aplicação no Elastic Beanstalk.

---

## Autores
- [Nicolas Ferreira](https://github.com/Niccofs)
- [Rafael Pereira](https://github.com/Kurokishin)
- [Tecla Fernandes](https://github.com/TeclaFernandes)
- [Samara Alcantara](https://github.com/SamaraAlcantara)
