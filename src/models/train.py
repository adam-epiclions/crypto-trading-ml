import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle
import matplotlib.pyplot as plt

class CryptoModelTrainer:
    def __init__(self, model_type='lstm'):
        """
        암호화폐 가격 예측 모델 학습기
        
        Args:
            model_type (str): 모델 유형 ('lstm', 'bilstm', 'gru')
        """
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def build_model(self, input_shape, output_size=1):
        """
        딥러닝 모델 구축
        
        Args:
            input_shape (tuple): 입력 데이터 형태 (시퀀스 길이, 특성 수)
            output_size (int): 출력 크기
            
        Returns:
            tf.keras.Model: 구축된 모델
        """
        model = Sequential()
        
        if self.model_type == 'lstm':
            model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=25))
            model.add(Dense(units=output_size))
            
        elif self.model_type == 'bilstm':
            model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(units=50, return_sequences=False)))
            model.add(Dropout(0.2))
            model.add(Dense(units=25))
            model.add(Dense(units=output_size))
            
        elif self.model_type == 'gru':
            model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(GRU(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=25))
            model.add(Dense(units=output_size))
            
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {self.model_type}")
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        모델 학습
        
        Args:
            X_train (np.array): 학습 입력 데이터
            y_train (np.array): 학습 타겟 데이터
            X_val (np.array): 검증 입력 데이터
            y_val (np.array): 검증 타겟 데이터
            epochs (int): 학습 에포크 수
            batch_size (int): 배치 크기
            
        Returns:
            tf.keras.callbacks.History: 학습 이력
        """
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        # 모델 저장 디렉토리 생성
        os.makedirs('models', exist_ok=True)
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=f'models/{self.model_type}_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # 검증 데이터가 제공되지 않은 경우 학습 데이터에서 분할
        if X_val is None or y_val is None:
            validation_split = 0.2
            validation_data = None
        else:
            validation_split = 0.0
            validation_data = (X_val, y_val)
        
        # 모델 학습
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        return history
    
    def save_model(self, model_path=None, history_path=None):
        """
        모델 및 학습 이력 저장
        
        Args:
            model_path (str): 모델 저장 경로
            history_path (str): 학습 이력 저장 경로
            
        Returns:
            tuple: (model_path, history_path)
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. 먼저 모델을 학습하세요.")
        
        if model_path is None:
            model_path = f'models/{self.model_type}_model.h5'
        
        if history_path is None:
            history_path = f'models/{self.model_type}_history.pkl'
        
        # 모델 저장
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"모델이 {model_path}에 저장되었습니다.")
        
        # 학습 이력 저장
        if self.history is not None:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
            print(f"학습 이력이 {history_path}에 저장되었습니다.")
        
        return model_path, history_path
    
    def plot_training_history(self):
        """
        학습 이력 시각화
        
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        if self.history is None:
            raise ValueError("시각화할 학습 이력이 없습니다. 먼저 모델을 학습하세요.")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.history.history['loss'], label='Train Loss')
        
        if 'val_loss' in self.history.history:
            ax.plot(self.history.history['val_loss'], label='Validation Loss')
        
        ax.set_title(f'{self.model_type.upper()} 모델 학습 이력')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # 그래프 저장
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/{self.model_type}_training_history.png')
        
        return fig

# 사용 예시
if __name__ == "__main__":
    # 가상의 데이터 생성 (실제로는 전처리된 데이터를 로드해야 함)
    X_train = np.random.random((1000, 60, 20))  # 1000개 샘플, 60 시퀀스 길이, 20 특성
    y_train = np.random.random(1000)  # 1000개 타겟 값
    
    # LSTM 모델 학습
    trainer = CryptoModelTrainer(model_type='lstm')
    trainer.build_model(input_shape=(60, 20))
    trainer.train(X_train, y_train, epochs=50)
    
    # 모델 저장
    trainer.save_model()
    
    # 학습 이력 시각화
    trainer.plot_training_history() 