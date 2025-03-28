from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load("naive_bayes_credit_approval_model.joblib")

mapeamento = {
    "Feminino": 0, "Masculino": 1,
    "Casado": 1, "Solteiro": 0, "Outro": 2,
    "Sim": 1, "Não": 0,
    "a": 0, "c": 1, "Ensino Médio": 2, "Ensino Médio incompleto": 3,
    "Ensino Fundamental incompleto": 4, "ff": 5,
    "Ensino superior incompleto": 6, "Ensino superior": 7,
    "Pós-Graduação Incompleta": 8, "Pós-Graduação": 9,
    "q": 10, "r": 11, "w": 12, "x": 13,
    "Cidadão": 0, "Residente permanente": 1, "Estrangeiro/temporário": 2,
    "Novo Cliente": 0, "Cliente Regular": 1, "Cliente VIP": 2
}

@app.route("/predict", methods=["POST"])
def prever():
    try:
        dados = request.json
  
        renda_categoria = int(float(dados["renda"]) // 1000)
    
        entrada = [
            mapeamento.get(str(dados["gênero"]), 0),
            float(dados["idade"]),
            float(dados["Dívida total existente"]),
            mapeamento.get(str(dados["estado civil"]), 0),
            mapeamento.get(str(dados["cliente do banco"]), 0),
            mapeamento.get(str(dados["nível de educação"]), 0),
            float(dados["quantidade de anos empregado"]),
            mapeamento.get(str(dados["histórico de inadimplência"]), 0),
            mapeamento.get(str(dados["empregado"]), 0),
            float(dados["CreditScore"]),
            mapeamento.get(str(dados["cidadão ou estrangeiro/temporário"]), 0),
            renda_categoria
        ]

        
        entrada_np = np.array(entrada).reshape(1, -1)
   
        previsao = modelo.predict(entrada_np)

        return jsonify({"previsao": int(previsao[0])})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
