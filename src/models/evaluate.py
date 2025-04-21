import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
from tensorflow.keras.models import load_model
import pickle

class CryptoModelEvaluator:
    def __init__(self, model_path, scaler_path=None):
        """
        암호화폐 예측 모델 평가기
        
        Args:
            model_path (str): 학습된 모델 파일 경로
            scaler_path (str): 저장된 스케일러 파일 경로 (선택 사항)
        """
        self.model = load_model(model_path)
        self.scaler = None
        
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def evaluate_model(self, X_test, y_test, feature_index=0):
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
        y_pred = self.model.predict(X_test)
        
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
        mse = mean_squared_error(y_test, y_pred.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred.flatten())
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        r2 = r2_score(y_test, y_pred.flatten())
        
        # 결과 저장
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2)
        }
        
        # 지표 출력
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.4f}%")
        print(f"R2 Score: {r2:.4f}")
        
        # 결과 저장
        os.makedirs('reports', exist_ok=True)
        with open('reports/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
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
    
    def plot_prediction_vs_actual(self, y_test, y_pred, title='예측 vs 실제'):
        """
        예측값과 실제값 비교 시각화
        
        Args:
            y_test (np.array): 실제 값
            y_pred (np.array): 예측 값
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 데이터 길이 확인 및 조정
        min_len = min(len(y_test), len(y_pred))
        y_test = y_test[:min_len]
        y_pred = y_pred[:min_len]
        
        # 인덱스 생성 (날짜가 없는 경우)
        x = np.arange(min_len)
        
        ax.plot(x, y_test, label='실제 가격', color='blue')
        ax.plot(x, y_pred, label='예측 가격', color='red', linestyle='--')
        
        ax.set_title(title)
        ax.set_xlabel('시간')
        ax.set_ylabel('가격')
        ax.legend()
        ax.grid(True)
        
        # 그래프 저장
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/prediction_vs_actual.png')
        
        return fig
    
    def plot_residuals(self, y_test, y_pred, title='잔차 분석'):
        """
        잔차 분석 시각화
        
        Args:
            y_test (np.array): 실제 값
            y_pred (np.array): 예측 값
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        # 잔차 계산
        residuals = y_test - y_pred
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 잔차 시계열 플롯
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('잔차 시계열')
        axes[0, 0].set_xlabel('시간')
        axes[0, 0].set_ylabel('잔차')
        axes[0, 0].axhline(y=0, color='r', linestyle='-')
        axes[0, 0].grid(True)
        
        # 잔차 히스토그램
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='blue')
        axes[0, 1].set_title('잔차 분포')
        axes[0, 1].set_xlabel('잔차')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].grid(True)
        
        # 예측값 대비 잔차 산점도
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 0].set_title('예측값 대비 잔차')
        axes[1, 0].set_xlabel('예측값')
        axes[1, 0].set_ylabel('잔차')
        axes[1, 0].axhline(y=0, color='r', linestyle='-')
        axes[1, 0].grid(True)
        
        # Q-Q 플롯
        from scipy import stats
        stats.probplot(residuals, plot=axes[1, 1])
        axes[1, 1].set_title('잔차 Q-Q 플롯')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.05)
        
        # 그래프 저장
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/residuals_analysis.png')
        
        return fig
    
    def plot_scatter_prediction_vs_actual(self, y_test, y_pred, title='예측 vs 실제 산점도'):
        """
        예측값과 실제값의 산점도 시각화
        
        Args:
            y_test (np.array): 실제 값
            y_pred (np.array): 예측 값
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 산점도 플롯
        ax.scatter(y_test, y_pred, alpha=0.5)
        
        # 이상적인 예측선 (y=x)
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(title)
        ax.set_xlabel('실제 가격')
        ax.set_ylabel('예측 가격')
        ax.grid(True)
        
        # 그래프 저장
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/scatter_prediction_vs_actual.png')
        
        return fig
    
    def generate_evaluation_report(self, X_test, y_test, feature_index=0):
        """
        종합 평가 보고서 생성
        
        Args:
            X_test (np.array): 테스트 입력 데이터
            y_test (np.array): 테스트 타겟 데이터
            feature_index (int): 타겟 특성의 인덱스
            
        Returns:
            dict: 평가 지표 및 그래프 경로
        """
        # 예측 수행
        y_pred = self.model.predict(X_test)
        
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
        metrics = self.evaluate_model(X_test, y_test, feature_index)
        
        # 시각화
        self.plot_prediction_vs_actual(y_test, y_pred.flatten())
        self.plot_residuals(y_test, y_pred.flatten())
        self.plot_scatter_prediction_vs_actual(y_test, y_pred.flatten())
        
        # 보고서 정보
        report_info = {
            'metrics': metrics,
            'figures': {
                'prediction_vs_actual': 'reports/figures/prediction_vs_actual.png',
                'residuals_analysis': 'reports/figures/residuals_analysis.png',
                'scatter_prediction_vs_actual': 'reports/figures/scatter_prediction_vs_actual.png'
            }
        }
        
        # 보고서 정보 저장
        with open('reports/evaluation_report.json', 'w') as f:
            json.dump(report_info, f, indent=4)
        
        print("평가 보고서가 생성되었습니다.")
        return report_info

# 사용 예시
if __name__ == "__main__":
    # 모델 로드 및 평가기 초기화
    evaluator = CryptoModelEvaluator(
        model_path='models/lstm_model.h5',
        scaler_path='models/scaler.pkl'
    )
    
    # 테스트 데이터 로드 (실제로는 전처리된 데이터를 로드해야 함)
    # 여기서는 가상의 데이터 생성
    X_test = np.random.random((100, 60, 20))  # 100개 샘플, 60 시퀀스 길이, 20 특성
    y_test = np.random.random(100) * 1000 + 5000  # 100개 타겟 값 (BTC 가격 범위로 가정)
    
    # 종합 평가 보고서 생성
    report_info = evaluator.generate_evaluation_report(X_test, y_test) 