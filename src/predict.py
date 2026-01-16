import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencia avisos do TensorFlow (CUDA, AVX2)

import pandas as pd
import joblib
from keras.models import load_model
from .config import Config

class StockPredictor:
    def __init__(self, model_type='neural_network'):
        self.model_type = model_type
        
        # Carregamento de artefatos
        try:
            self.scaler = joblib.load(Config.MODEL_DIR / "scaler.pkl")
            if model_type == 'neural_network':
                self.model = load_model(Config.MODEL_DIR / "neural_network_model.keras")
            else:
                self.model = joblib.load(Config.MODEL_DIR / "linear_regression_model.pkl")
            
            self.encoders = {}
            for col in Config.COLS_CATEGORICAS:
                self.encoders[col] = joblib.load(Config.MODEL_DIR / f"encoder_{col}.pkl")
        except Exception as e:
            print(f"❌ Erro ao carregar modelos/artefatos: {e}")
            raise

    def _prepare_input(self, data_dict):
        """Transforma um dicionário de entrada no formato que o modelo espera."""
        df = pd.DataFrame([data_dict])
        
        # Encoding Categórico com tratamento de erro amigável
        for col, le in self.encoders.items():
            val = str(df[col].iloc[0]).strip()
            
            if val not in le.classes_:
                raise ValueError(
                    f"\n❌ Erro na coluna '{col}': O valor '{val}' é desconhecido.\n"
                    f"💡 Valores aceitos pelo seu modelo: {list(le.classes_)}"
                )
            
            df[col] = le.transform([val])
            
        # Garante a ordem exata das colunas utilizada no treinamento
        cols_features = [c for c in Config.COLUNAS_INTERESSE if c != Config.TARGET]
        df = df[cols_features]
        
        # Scaling
        return self.scaler.transform(df.values)

    def predict(self, input_data):
        prepared_data = self._prepare_input(input_data)
        prediction = self.model.predict(prepared_data, verbose=0)
        
        # Retorno tratado conforme o tipo de modelo
        return float(prediction[0][0]) if self.model_type == 'neural_network' else float(prediction[0])

if __name__ == "__main__":
    # Exemplo de uso com dados fictícios de uma ação
    sample_input = {
        'sectorCompany': 'BM&FBOVESPA FINANCIALS INDEX (IFNC)',
        'segmentCompany': 'SEGMENTS AND SECTORS',
        'dayTime': 15, 
        'dayWeekTime': 4, 
        'monthTime': 1, 
        'yearTime': 2026,
        'openValueStock': 34.50, 
        'highValueStock': 35.10, 
        'lowValueStock': 34.20, 
        'quantityStock': 1500000,
        'valueCoin': 5.45
    }

    try:
        predictor = StockPredictor(model_type='neural_network')
        result = predictor.predict(sample_input)
        
        print(f"\n✅ --- Resultado da Predição ({predictor.model_type}) ---")
        print(f"💰 Preço de Fechamento Estimado: R$ {result:.2f}")
    except Exception as e:
        print(e)