import pandas as pd
import requests
import ccxt
import os
from datetime import datetime, timedelta

class CryptoDataCollector:
    def __init__(self, exchange='binance', symbol='BTC/USDT', timeframe='1h'):
        """
        암호화폐 거래 데이터 수집기
        
        Args:
            exchange (str): 거래소 이름 (binance, coinbase 등)
            symbol (str): 거래 심볼 (BTC/USDT 등)
            timeframe (str): 시간 프레임 (1m, 5m, 1h, 1d 등)
        """
        self.exchange = getattr(ccxt, exchange)()
        self.symbol = symbol
        self.timeframe = timeframe
        
    def fetch_ohlcv(self, start_date, end_date=None):
        """
        OHLCV (Open, High, Low, Close, Volume) 데이터 수집
        
        Args:
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD), 기본값은 현재
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        ohlcv_data = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            try:
                data = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=current_timestamp,
                    limit=1000  # 대부분의 거래소는 한 번에 최대 1000개 제공
                )
                
                if not data:
                    break
                    
                ohlcv_data.extend(data)
                
                # 마지막 데이터의 타임스탬프 + 1ms를 다음 시작점으로 설정
                current_timestamp = data[-1][0] + 1
                
                # API 호출 제한을 피하기 위한 대기
                self.exchange.sleep(self.exchange.rateLimit)
                
            except Exception as e:
                print(f"데이터 수집 중 오류 발생: {e}")
                break
        
        # 데이터프레임으로 변환
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def save_data(self, df, filename=None):
        """
        수집된 데이터를 파일로 저장
        
        Args:
            df (pd.DataFrame): 저장할 데이터프레임
            filename (str): 파일 이름, 기본값은 exchange_symbol_timeframe_date.csv
        """
        if filename is None:
            symbol_name = self.symbol.replace('/', '_')
            today = datetime.now().strftime('%Y%m%d')
            filename = f"{self.exchange.id}_{symbol_name}_{self.timeframe}_{today}.csv"
        
        # 디렉토리 생성
        os.makedirs('data/raw', exist_ok=True)
        filepath = os.path.join('data/raw', filename)
        
        # CSV 파일로 저장
        df.to_csv(filepath)
        print(f"데이터가 {filepath}에 저장되었습니다.")
        
        return filepath

# 사용 예시
if __name__ == "__main__":
    collector = CryptoDataCollector(exchange='binance', symbol='BTC/USDT', timeframe='1h')
    
    # 최근 30일 데이터 수집
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    df = collector.fetch_ohlcv(start_date)
    
    # 데이터 저장
    collector.save_data(df) 