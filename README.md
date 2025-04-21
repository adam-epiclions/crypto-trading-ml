# 암호화폐 가격 예측 딥러닝 프로젝트

이 프로젝트는 딥러닝 모델을 사용하여 암호화폐 가격을 예측하는 종합적인 파이프라인을 구현합니다. 데이터 수집부터 전처리, 모델 학습, 예측 및 평가, 시각화까지 전체 과정을 포함합니다.

## 프로젝트 구조 
crypto-trading-ml/
├── data/
│ ├── raw/ # 원본 데이터 저장
│ ├── processed/ # 전처리된 데이터
│ └── features/ # 특성 엔지니어링 결과
├── notebooks/ # 탐색적 데이터 분석 및 모델 실험
├── src/
│ ├── data/ # 데이터 수집 및 처리
│ │ ├── init.py
│ │ ├── collect.py # API 연동 및 데이터 수집
│ │ └── preprocess.py # 데이터 전처리
│ ├── features/ # 특성 엔지니어링
│ │ ├── init.py
│ │ └── build_features.py
│ ├── models/ # 모델 구현
│ │ ├── init.py
│ │ ├── train.py # 모델 학습
│ │ ├── predict.py # 예측 실행
│ │ └── evaluate.py # 모델 평가
│ └── visualization/ # 시각화
│ ├── init.py
│ └── visualize.py
├── models/ # 저장된 모델 파일
├── reports/ # 결과 보고서
│ ├── figures/ # 생성된 그래프
│ └── html/ # 인터랙티브 시각화
├── config/ # 설정 파일
├── requirements.txt # 의존성 패키지
└── README.md # 프로젝트 설명
```

## 주요 기능

### 1. 데이터 수집 (`src/data/collect.py`)
- 다양한 암호화폐 거래소(Binance, Coinbase 등)에서 OHLCV 데이터 수집
- 특정 기간의 데이터를 CSV 파일로 저장
- ccxt 라이브러리를 사용하여 여러 거래소 API 지원

### 2. 데이터 전처리 (`src/data/preprocess.py`)
- 기술적 지표 추가:
  - 이동평균선(MA7, MA14, MA30)
  - 상대강도지수(RSI)
  - 볼린저 밴드
  - MACD(Moving Average Convergence Divergence)
- 데이터 정규화
- 시계열 시퀀스 생성
- 학습/테스트 데이터 분할

### 3. 모델 학습 (`src/models/train.py`)
- 다양한 딥러닝 모델 구현:
  - LSTM(Long Short-Term Memory)
  - BiLSTM(Bidirectional LSTM)
  - GRU(Gated Recurrent Unit)
- 조기 종료, 모델 체크포인트 등 학습 최적화
- 학습 이력 시각화 및 저장

### 4. 예측 및 평가 (`src/models/predict.py`, `src/models/evaluate.py`)
- 학습된 모델을 사용한 가격 예측
- 다양한 평가 지표 계산:
  - MSE(Mean Squared Error)
  - RMSE(Root Mean Squared Error)
  - MAE(Mean Absolute Error)
  - MAPE(Mean Absolute Percentage Error)
  - R2 Score
- 예측 결과 시각화
- 미래 가격 예측

### 5. 시각화 (`src/visualization/visualize.py`)
- 가격 추이, 캔들스틱 차트
- 기술적 지표 시각화
- 상관관계 분석
- 거래량 분석
- 수익률 분포
- 변동성 분석
- 인터랙티브 대시보드
- 계절성 분석

## 설치 및 사용 방법

### 설치

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/crypto-trading-ml.git
cd crypto-trading-ml
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

### 사용 예시

#### 1. 데이터 수집
```python
from src.data.collect import CryptoDataCollector

collector = CryptoDataCollector(exchange='binance', symbol='BTC/USDT', timeframe='1h')
df = collector.fetch_ohlcv(start_date='2023-01-01', end_date='2023-12-31')
collector.save_data(df)
```

#### 2. 데이터 전처리
```python
from src.data.preprocess import CryptoDataPreprocessor

preprocessor = CryptoDataPreprocessor()
df = preprocessor.load_data('data/raw/binance_BTC_USDT_1h_20231231.csv')
df = preprocessor.add_technical_indicators(df)
df_normalized = preprocessor.normalize_data(df)
X, y = preprocessor.create_sequences(df_normalized)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)
```

#### 3. 모델 학습
```python
from src.models.train import CryptoModelTrainer

trainer = CryptoModelTrainer(model_type='lstm')
trainer.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
trainer.train(X_train, y_train, X_test, y_test, epochs=100)
trainer.save_model()
```

#### 4. 예측 및 평가
```python
from src.models.predict import CryptoPredictor
from src.models.evaluate import CryptoModelEvaluator

predictor = CryptoPredictor('models/lstm_model.h5', 'models/scaler.pkl')
predictions = predictor.predict(X_test)
predictor.plot_predictions(y_test, predictions.flatten())

evaluator = CryptoModelEvaluator('models/lstm_model.h5', 'models/scaler.pkl')
report = evaluator.generate_evaluation_report(X_test, y_test)
```

#### 5. 결과 시각화
```python
from src.visualization.visualize import CryptoVisualizer

visualizer = CryptoVisualizer()
visualizer.plot_price_history(df)
visualizer.plot_technical_indicators(df)
visualizer.plot_interactive_dashboard(df)
```

## 의존성 패키지

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- ccxt

## 향후 개발 계획

- 추가 모델 구현 (Transformer, CNN-LSTM 등)
- 포트폴리오 최적화 기능
- 트레이딩 전략 백테스팅
- 실시간 예측 및 알림 시스템
- 웹 인터페이스 개발

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.