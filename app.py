import logging
from flask import Flask, request, jsonify, make_response
import joblib
import pandas as pd
from flask_cors import CORS
from category_encoders.one_hot import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

CORS(app)  # Habilitar CORS para todas as rotas

# Configuração de logging
logging.basicConfig(level=logging.INFO)

# Carregando o modelo e o encoder salvo
model = joblib.load('melhor_modelo.pkl') 

# Carregar o conjunto de dados de treinamento
X_train = joblib.load('dados_treino.pkl')


# Preparar colunas para transformação
num_cols_for_transform = ['idade', 'hipertensao', 'doenca_cardiaca', 'nivel_glicose', 'imc']
cat_cols_for_transform = ['casado', 'tipo_trabalho', 'tipo_residencia', 'condicao_fumante']

# Pipeline para colunas numéricas
num_pipe = make_pipeline(StandardScaler())

# Pipeline para colunas categóricas
cat_pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"))

# Criação do pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols_for_transform),
        ("cat", cat_pipe, cat_cols_for_transform),
    ]
)
@app.get('/')
def index():
    return 'app teste'

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar se o tipo de conteúdo é JSON
    if request.content_type != 'application/json':
        return make_response(jsonify({'error': 'Content-Type must be application/json'}), 415)
    
    # Obter os dados do pedido
    try:
        dados_json = request.json
        logging.info(f'Received data: {dados_json}')
        dados = pd.DataFrame([dados_json])
    except Exception as e:
        return make_response(jsonify({'error': 'Invalid JSON data', 'message': str(e)}), 400)
    
    # Verificar se há dados no JSON
    if dados.empty:
        return make_response(jsonify({'error': 'Empty JSON data'}), 400)

    # Transformar os dados recebidos
    try:
        X_train_transform = preprocessor.fit_transform(X_train)
        X = preprocessor.transform(dados)
        X = pd.DataFrame(X)
        print(X)
    except Exception as e:
        return make_response(jsonify({'error': 'Data transformation error', 'message': str(e)}), 400)

    # Fazer a previsão
    try:
        predicoes = model.predict(X)
    except Exception as e:
        return make_response(jsonify({'error': 'Prediction error', 'message': str(e)}), 500)

    # Retornar as previsões como JSON
    return jsonify(predicoes.tolist())

if __name__ == '__main__':
    app.run()
