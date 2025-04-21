import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import pickle
import json

class CryptoPredictor:
    def __init__(self, model_path, scaler_path=None):
        """
        암호화폐 가격 예측기
        
        Args:
            model_path (str): 학습된 모델 파일 경로
            scaler_path (str): 저장된 스케일러 파일 경로 (선택 사항)
        """
        self.model = load_model(model_path)
        self.scaler = None
        
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def predict(self, X):
        """
        가격 예측 수행
        
        Args:
            X (np.array): 입력 시퀀스 데이터
            
        Returns:
            np.array: 예측 결과
        """
        predictions = self.model.predict(X)
        return predictions
    
    def inverse_transform(self, predictions, feature_index=0):
        """
        정규화된 예측값을 원래 스케일로 변환
        
        Args:
            predictions (np.array): 정규화된 예측값
            feature_index (int): 타겟 특성의 인덱스
            
        Returns:
            np.array: 원래 스케일로 변환된 예측값
        """
        if self.scaler is None:
            print("경고: 스케일러가 제공되지 않았습니다. 원본 예측값을 반환합니다.")
            return predictions
        
        # 스케일러가 다차원 데이터에 맞게 설정되어 있으므로 변환을 위한 더미 배열 생성
        dummy = np.zeros((len(predictions), self.scaler.scale_.shape[0]))
        dummy[:, feature_index] = predictions.flatten()
        
        # 역변환 수행
        dummy = self.scaler.inverse_transform(dummy)
        
        # 타겟 특성만 추출
        return dummy[:, feature_index]
    
    def plot_predictions(self, y_true, y_pred, title='가격 예측 결과'):
        """
        예측 결과 시각화
        
        Args:
            y_true (np.array): 실제 값
            y_pred (np.array): 예측 값
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 데이터 길이 확인 및 조정
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # 인덱스 생성 (날짜가 없는 경우)
        x = np.arange(min_len)
        
        ax.plot(x, y_true, label='실제 가격', color='blue')
        ax.plot(x, y_pred, label='예측 가격', color='red', linestyle='--')
        
        ax.set_title(title)
        ax.set_xlabel('시간')
        ax.set_ylabel('가격')
        ax.legend()
        ax.grid(True)
        
        # 그래프 저장
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/prediction_results.png')
        
        return fig
    
    def evaluate(self, X_test, y_test, feature_index=0):
        """
        모델 성능 평가
        
        Args:
            X_test (np.array): 테스트 입력 데이터
            y_test (np.array): 테스트 타겟 데이터
            feature_index (int): 타겟 특성의 인덱스
            
        Returns:
            dict: 평가 지표
        """
        # 예측 수행
        y_pred = self.predict(X_test)
        
        # 필요한 경우 원래 스케일로 변환
        if self.scaler is not None:
            y_pred = self.inverse_transform(y_pred, feature_index)
            
            # y_test가 정규화되어 있다면 역변환
            if np.max(y_test) <= 1 and np.min(y_test) >= 0:
                dummy = np.zeros((len(y_test), self.scaler.scale_.shape[0]))
                dummy[:, feature_index] = y_test.flatten()
                dummy = self.scaler.inverse_transform(dummy)
                y_test = dummy[:, feature_index]
        
        # 평가 지표 계산
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        
        # 결과 저장
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape)
        }
        
        # 지표 출력
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.4f}%")
        
        # 결과 저장
        os.makedirs('reports', exist_ok=True)
        with open('reports/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 예측 결과 시각화
        self.plot_predictions(y_test, y_pred.flatten(), title='테스트 데이터 예측 결과')
        
        return metrics
    
    def predict_future(self, last_sequence, steps=30, feature_index=0):
        """
        미래 가격 예측
        
        Args:
            last_sequence (np.array): 마지막 입력 시퀀스
            steps (int): 예측할 미래 시점 수
            feature_index (int): 타겟 특성의 인덱스
            
        Returns:
            np.array: 미래 예측값
        """
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # 현재 시퀀스로 다음 값 예측
            pred = self.model.predict(np.expand_dims(current_sequence, axis=0))
            future_predictions.append(pred[0, 0])
            
            # 시퀀스 업데이트 (가장 오래된 값 제거하고 예측값 추가)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            
            # 마지막 행의 타겟 특성 위치에 예측값 삽입
            current_sequence[-1, feature_index] = pred[0, 0]
        
        # 예측값을 원래 스케일로 변환
        if self.scaler is not None:
            future_predictions = self.inverse_transform(
                np.array(future_predictions).reshape(-1, 1), 
                feature_index
            )
        
        return np.array(future_predictions)
    
    def plot_future_predictions(self, historical_data, future_predictions, date_index=None, title='미래 가격 예측'):
        """
        미래 예측 결과 시각화
        
        Args:
            historical_data (np.array): 과거 실제 데이터
            future_predictions (np.array): 미래 예측 데이터
            date_index (pd.DatetimeIndex): 날짜 인덱스 (선택 사항)
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 날짜 인덱스가 제공되지 않은 경우 숫자 인덱스 사용
        if date_index is None:
            x_hist = np.arange(len(historical_data))
            x_future = np.arange(len(historical_data), len(historical_data) + len(future_predictions))
        else:
            # 날짜 인덱스 사용
            x_hist = date_index[:len(historical_data)]
            
            # 미래 날짜 생성 (마지막 날짜부터 예측 기간만큼)
            last_date = date_index[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(date_index)
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                            periods=len(future_predictions), 
                                            freq=freq)
            else:
                # 날짜가 아닌 경우 숫자 인덱스로 대체
                future_dates = np.arange(len(historical_data), len(historical_data) + len(future_predictions))
            
            x_future = future_dates
        
        # 과거 데이터 플롯
        ax.plot(x_hist, historical_data, label='과거 실제 가격', color='blue')
        
        # 미래 예측 플롯
        ax.plot(x_future, future_predictions, label='미래 예측 가격', color='red', linestyle='--')
        
        # 현재 시점 표시 (과거와 미래의 경계)
        if date_index is None:
            ax.axvline(x=len(historical_data)-1, color='green', linestyle='-', alpha=0.7, label='현재')
        else:
            ax.axvline(x=date_index[-1], color='green', linestyle='-', alpha=0.7, label='현재')
        
        ax.set_title(title)
        ax.set_xlabel('시간')
        ax.set_ylabel('가격')
        ax.legend()
        ax.grid(True)
        
        # 그래프 저장
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/future_predictions.png')
        
        return fig
    
    def save_predictions(self, predictions, filename='future_predictions.csv', dates=None):
        """
        예측 결과를 파일로 저장
        
        Args:
            predictions (np.array): 예측 결과
            filename (str): 저장할 파일 이름
            dates (pd.DatetimeIndex): 날짜 인덱스 (선택 사항)
            
        Returns:
            str: 저장된 파일 경로
        """
        os.makedirs('reports/predictions', exist_ok=True)
        filepath = os.path.join('reports/predictions', filename)
        
        if dates is not None and len(dates) == len(predictions):
            df = pd.DataFrame({'date': dates, 'prediction': predictions})
            df.set_index('date', inplace=True)
        else:
            df = pd.DataFrame({'prediction': predictions})
        
        df.to_csv(filepath)
        print(f"예측 결과가 {filepath}에 저장되었습니다.")
        
        return filepath

# 사용 예시
if __name__ == "__main__":
    # 모델 로드 및 예측기 초기화
    predictor = CryptoPredictor(
        model_path='models/lstm_model.h5',
        scaler_path='models/scaler.pkl'
    )
    
    # 테스트 데이터 로드 (실제로는 전처리된 데이터를 로드해야 함)
    # 여기서는 가상의 데이터 생성
    X_test = np.random.random((100, 60, 20))  # 100개 샘플, 60 시퀀스 길이, 20 특성
    y_test = np.random.random(100) * 1000 + 5000  # 100개 타겟 값 (BTC 가격 범위로 가정)
    
    # 모델 평가
    metrics = predictor.evaluate(X_test, y_test)
    
    # 미래 예측
    last_sequence = X_test[-1]
    future_predictions = predictor.predict_future(last_sequence, steps=30)
    
    # 미래 예측 시각화
    predictor.plot_future_predictions(y_test, future_predictions, title='비트코인 30일 가격 예측')
    
    # 예측 결과 저장
    predictor.save_predictions(future_predictions) 