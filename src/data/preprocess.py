import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class CryptoDataPreprocessor:
    def __init__(self):
        """
        암호화폐 데이터 전처리기
        """
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath):
        """
        데이터 파일 로드
        
        Args:
            filepath (str): 데이터 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    
    def add_technical_indicators(self, df):
        """
        기술적 지표 추가
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터프레임
        """
        # 이동평균선 (Moving Average)
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA14'] = df['close'].rolling(window=14).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        
        # 상대강도지수 (Relative Strength Index, RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드 (Bollinger Bands)
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # MACD (Moving Average Convergence Divergence)
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # 결측치 제거
        df.dropna(inplace=True)
        
        return df
    
    def normalize_data(self, df, columns=None):
        """
        데이터 정규화
        
        Args:
            df (pd.DataFrame): 정규화할 데이터프레임
            columns (list): 정규화할 열 목록, None이면 모든 수치형 열
            
        Returns:
            pd.DataFrame: 정규화된 데이터프레임
        """
        if columns is None:
            # 수치형 열만 선택
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 정규화 수행
        df_normalized = df.copy()
        df_normalized[columns] = self.scaler.fit_transform(df[columns])
        
        return df_normalized
    
    def create_sequences(self, df, target_col='close', seq_length=60):
        """
        시계열 시퀀스 생성
        
        Args:
            df (pd.DataFrame): 데이터프레임
            target_col (str): 예측 대상 열
            seq_length (int): 시퀀스 길이 (과거 몇 개의 데이터를 사용할지)
            
        Returns:
            tuple: (X, y) - 입력 시퀀스와 타겟 값
        """
        X, y = [], []
        
        for i in range(len(df) - seq_length):
            X.append(df.iloc[i:i+seq_length].values)
            y.append(df.iloc[i+seq_length][target_col])
        
        return np.array(X), np.array(y)
    
    def train_test_split(self, X, y, test_size=0.2):
        """
        학습 및 테스트 데이터 분할
        
        Args:
            X (np.array): 입력 데이터
            y (np.array): 타겟 데이터
            test_size (float): 테스트 데이터 비율
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, df, filename):
        """
        전처리된 데이터 저장
        
        Args:
            df (pd.DataFrame): 저장할 데이터프레임
            filename (str): 파일 이름
            
        Returns:
            str: 저장된 파일 경로
        """
        os.makedirs('data/processed', exist_ok=True)
        filepath = os.path.join('data/processed', filename)
        
        df.to_csv(filepath)
        print(f"전처리된 데이터가 {filepath}에 저장되었습니다.")
        
        return filepath

# 사용 예시
if __name__ == "__main__":
    preprocessor = CryptoDataPreprocessor()
    
    # 데이터 로드
    df = preprocessor.load_data('data/raw/binance_BTC_USDT_1h_20230101.csv')
    
    # 기술적 지표 추가
    df = preprocessor.add_technical_indicators(df)
    
    # 데이터 정규화
    df_normalized = preprocessor.normalize_data(df)
    
    # 전처리된 데이터 저장
    preprocessor.save_processed_data(df_normalized, 'processed_btc_usdt_1h.csv')
    
    # 시퀀스 생성
    X, y = preprocessor.create_sequences(df_normalized)
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)
    
    print(f"학습 데이터 형태: {X_train.shape}, {y_train.shape}")
    print(f"테스트 데이터 형태: {X_test.shape}, {y_test.shape}") 