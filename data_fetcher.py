import requests
import pandas as pd
from datetime import datetime, timedelta

class BithumbDataFetcher:
    def __init__(self):
        self.base_url = "https://api.bithumb.com/public"
    
    def get_orderbook(self, symbol):
        """호가 데이터 조회 테스트"""
        url = f"{self.base_url}/orderbook/{symbol}_KRW"
        response = requests.get(url)
        data = response.json()['data']
        
        return {
            'asks': data['asks'],
            'bids': data['bids'],
            'timestamp': data['timestamp']
        }
    
    def get_candlestick(self, symbol, interval='1m', count=200):
        """분봉 데이터 조회
        interval: 1m, 3m, 5m, 10m, 30m, 1h, 6h, 12h, 24h
        """
        url = f"{self.base_url}/candlestick/{symbol}_KRW/{interval}"
        response = requests.get(url)
        data = response.json()['data']
        
        df = pd.DataFrame(data, columns=['time', 'open', 'close', 'high', 'low', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df.sort_values('time')
    
    def get_recent_trades(self, symbol, count=100):
        """최근 체결 내역 조회"""
        url = f"{self.base_url}/transaction_history/{symbol}_KRW"
        response = requests.get(url)
        data = response.json()['data']
        
        return pd.DataFrame(data)

def analyze_trading_opportunity(orderbook):
    """ADA 거래 기회 분석"""
    asks = orderbook['asks']
    bids = orderbook['bids']
    
    # 1. 스프레드 계산
    best_ask = float(asks[0]['price'])
    best_bid = float(bids[0]['price'])
    spread = best_ask - best_bid
    spread_percent = (spread / best_bid) * 100
    
    # 2. 매수/매도 물량 비율 계산 (상위 5개 호가로 수정)
    ask_volume = sum(float(ask['quantity']) for ask in asks[:5])
    bid_volume = sum(float(bid['quantity']) for bid in bids[:5])
    volume_ratio = bid_volume / ask_volume if ask_volume > 0 else 0
    
    # 3. 거래 기회 판단 (실제 데이터 기반으로 수정)
    is_tradeable = (
        spread_percent <= 0.37 and  # 실제 평균 스프레드
        0.7 <= volume_ratio <= 1.3   # 실제 데이터 기반 범위
    )
    
    return {
        'current_price': best_bid,
        'spread': spread,
        'spread_percent': spread_percent,
        'volume_ratio': volume_ratio,
        'is_tradeable': is_tradeable,
        'best_ask': best_ask,
        'best_bid': best_bid
    }

def analyze_liquidity(orderbook):
    """호가별 유동성 분석"""
    print("\n=== 호가 유동성 분석 ===")
    
    # 매도/매수 호가 상위 5개 분석
    print("\n매도 호가:")
    total_ask_volume = 0
    for ask in orderbook['asks'][:5]:
        price = float(ask['price'])
        quantity = float(ask['quantity'])
        amount = price * quantity
        total_ask_volume += amount
        print(f"가격: {price}원 - 수량: {quantity:,.0f}개 - 금액: {amount:,.0f}원")
    
    print(f"\n매도 5호가 총액: {total_ask_volume:,.0f}원")
    
    print("\n매수 호가:")
    total_bid_volume = 0
    for bid in orderbook['bids'][:5]:
        price = float(bid['price'])
        quantity = float(bid['quantity'])
        amount = price * quantity
        total_bid_volume += amount
        print(f"가격: {price}원 - 수량: {quantity:,.0f}개 - 금액: {amount:,.0f}원")
    
    print(f"\n매수 5호가 총액: {total_bid_volume:,.0f}원")
    
    # 적정 거래 금액 추천
    avg_volume = (total_ask_volume + total_bid_volume) / 10  # 평균 호가 금액
    
    print(f"\n=== 거래 금액 추천 ===")
    print(f"호가당 평균 금액: {avg_volume:,.0f}원")
    print(f"권장 최소 거래 금액: {avg_volume * 0.01:,.0f}원 (평균의 1%)")
    print(f"권장 최대 거래 금액: {avg_volume * 0.1:,.0f}원 (평균의 10%)")

# 테스트 코드
if __name__ == "__main__":
    fetcher = BithumbDataFetcher()
    
    try:
        print("\n=== ADA 거래 기회 분석 ===")
        orderbook = fetcher.get_orderbook("ADA")
        analysis = analyze_trading_opportunity(orderbook)
        
        print(f"현재 가격: {analysis['current_price']}원")
        print(f"스프레드: {analysis['spread']}원 ({analysis['spread_percent']:.3f}%)")
        print(f"매수/매도 물량 비율: {analysis['volume_ratio']:.2f}")
        print(f"거래 가능 여부: {'예' if analysis['is_tradeable'] else '아니오'}")
        
        if analysis['is_tradeable']:
            print("\n=== 거래 전략 ===")
            entry_price = analysis['best_bid']  # 매수 호가에 진입
            target_price = entry_price * 1.0012  # 0.12% 수익 목표
            stop_loss = entry_price * 0.9992    # -0.08% 손절
            
            print(f"진입 가격: {entry_price}원")
            print(f"목표 가격: {target_price:.2f}원 (+0.12%)")
            print(f"손절 가격: {stop_loss:.2f}원 (-0.08%)")
            
            # 적정 거래량 계산 (최소 주문금액 5000원 고려)
            min_quantity = max(5000 / entry_price, 10)  # 최소 10개 이상
            print(f"최소 거래량: {min_quantity:.0f}개")
            
        analyze_liquidity(orderbook)
        
    except Exception as e:
        print(f"에러 발생: {e}") 