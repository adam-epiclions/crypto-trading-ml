import pandas as pd
import time
from datetime import datetime, timedelta
from data_fetcher import BithumbDataFetcher
import numpy as np

class ADALiveBacktest:
    def __init__(self):
        self.fetcher = BithumbDataFetcher()
        self.holdings = 0
        
        # 소액 거래 설정
        self.MIN_TRADE_KRW = 5000    # 최소 거래금액 5천원
        self.MAX_TRADE_KRW = 10000   # 최대 거래금액 1만원
        self.MAX_TOTAL_KRW = 100000  # 총 투자한도 10만원
        
        # 거래 조건 설정
        self.SPREAD_THRESHOLD = 0.37
        self.VOLUME_RATIO_MIN = 0.7
        self.VOLUME_RATIO_MAX = 1.3
        self.trades = []
        self.price_history = []
        self.RSI_PERIOD = 14
        self.AVG_ORDERS = 3.66  # 평균 주문수
        self.MAX_ORDERS = 15    # 최대 주문수
        
    def check_trading_condition(self, orderbook, rsi):
        """거래 조건 확인"""
        asks = orderbook['asks']
        bids = orderbook['bids']
        
        best_ask = float(asks[0]['price'])
        best_bid = float(bids[0]['price'])
        spread = (best_ask - best_bid) / best_bid * 100
        
        # 상위 5개 호가 기준으로 수정
        ask_volume = sum(float(ask['quantity']) for ask in asks[:5])
        bid_volume = sum(float(bid['quantity']) for bid in bids[:5])
        volume_ratio = bid_volume / ask_volume if ask_volume > 0 else 0
        
        # RSI 구간별 실제 성공률 기반 주문 크기 결정
        if rsi <= 30:  # 과매도
            if volume_ratio > 1:  # upper band (100% 성공률)
                order_size = float(bids[0]['quantity']) * 1.0
            else:  # middle band (71.9% 성공률)
                order_size = float(bids[0]['quantity']) * 0.72
        elif rsi >= 70:  # 과매수
            if volume_ratio < 1:  # lower band (58.7% 성공률)
                order_size = float(bids[0]['quantity']) * 0.59
            else:  # middle band (78.9% 성공률)
                order_size = float(bids[0]['quantity']) * 0.79
        else:  # 중립 (50~58.7% 성공률)
            if volume_ratio < 0.9:  # lower band
                order_size = float(bids[0]['quantity']) * 0.50
            elif volume_ratio > 1.1:  # upper band
                order_size = float(bids[0]['quantity']) * 0.59
            else:  # middle band
                order_size = float(bids[0]['quantity']) * 0.54
        
        # 최소/최대 주문 크기 제한
        order_size = min(max(order_size, 0), float(bids[0]['quantity']))
        
        return {
            'is_tradeable': (
                spread <= self.SPREAD_THRESHOLD and
                self.VOLUME_RATIO_MIN <= volume_ratio <= self.VOLUME_RATIO_MAX and
                order_size > 0 and
                self.holdings * best_bid < self.MAX_TOTAL_KRW
            ),
            'current_price': best_bid,
            'spread': spread,
            'volume_ratio': volume_ratio,
            'order_size': order_size,
            'rsi': rsi,
            'total_invested': self.holdings * best_bid
        }
    
    def close_all_positions(self):
        """보유 물량 전량 청산"""
        if self.holdings > 0:
            print("\n=== 보유 물량 청산 중 ===")
            print(f"청산 수량: {self.holdings:,.0f} ADA")
            
            orderbook = self.fetcher.get_orderbook("ADA")
            current_price = float(orderbook['bids'][0]['price'])
            
            liquidation_amount = self.holdings * current_price
            print(f"청산 금액: {liquidation_amount:,.0f}원")
            print(f"청산 가격: {current_price:,.2f}원")
            
            self.trades.append({
                'timestamp': datetime.now(),
                'type': 'liquidation',
                'price': current_price,
                'quantity': self.holdings,
                'amount': liquidation_amount
            })
            
            self.holdings = 0
            return liquidation_amount
    
    def calculate_rsi(self):
        """RSI 계산"""
        if len(self.price_history) < self.RSI_PERIOD + 1:
            return 50
        
        deltas = np.diff(self.price_history)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gain[:self.RSI_PERIOD])
        avg_loss = np.mean(loss[:self.RSI_PERIOD])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def run_live_test(self, duration_minutes=60):
        print(f"=== ADA 실제 거래 테스트 시작 ({duration_minutes}분) ===")
        print("소액 거래 설정:")
        print(f"최소 거래금액: {self.MIN_TRADE_KRW:,}원")
        print(f"최대 거래금액: {self.MAX_TRADE_KRW:,}원")
        print(f"총 투자한도: {self.MAX_TOTAL_KRW:,}원")
        print("실시간 호가 데이터 기반 거래 설정:")
        print(f"평균 주문수: {self.AVG_ORDERS:.2f}")
        print(f"최대 주문수: {self.MAX_ORDERS}")
        print(f"과매도(RSI<30): 평균 주문수의 120%")
        print(f"중립: 평균 주문수")
        print(f"과매수(RSI>70): 평균 주문수의 80%")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        total_trades = 0
        successful_trades = 0
        
        try:
            while datetime.now() < end_time:
                # 실시간 데이터 수집
                orderbook = self.fetcher.get_orderbook("ADA")
                current_candle = self.fetcher.get_candlestick("ADA", interval='1m', count=1)
                
                # RSI 계산 (실제 구현 필요)
                current_price = float(orderbook['bids'][0]['price'])
                self.price_history.append(current_price)
                if len(self.price_history) > self.RSI_PERIOD + 1:
                    self.price_history.pop(0)
                rsi = self.calculate_rsi()
                
                conditions = self.check_trading_condition(orderbook, rsi)
                
                print(f"\n[{datetime.now()}] 시장 상황:")
                print(f"현재가: {conditions['current_price']:,}원")
                print(f"스프레드: {conditions['spread']:.3f}%")
                print(f"매수/매도 비율: {conditions['volume_ratio']:.2f}")
                print(f"RSI: {conditions['rsi']:.2f}")
                print(f"현재 보유량: {self.holdings:,} ADA")
                
                if conditions['is_tradeable']:
                    order_size = conditions['order_size']
                    order_amount = order_size * conditions['current_price']
                    
                    print("\n거래 신호 발생!")
                    print(f"주문 수량: {order_size:,.2f} ADA")
                    print(f"주문 금액: {order_amount:,.0f}원")
                    
                    # 거래 기록
                    self.trades.append({
                        'timestamp': datetime.now(),
                        'type': 'buy',
                        'price': conditions['current_price'],
                        'quantity': order_size,
                        'amount': order_amount,
                        'rsi': conditions['rsi']
                    })
                    
                    self.holdings += order_size
                    total_trades += 1
                    
                time.sleep(60)  # 1분 대기
                
        except KeyboardInterrupt:
            print("\n프로그램 종료 요청됨")
            liquidation_amount = self.close_all_positions()
            
        except Exception as e:
            print(f"\n에러 발생: {e}")
            self.close_all_positions()
            
        finally:
            print("\n=== 최종 결과 ===")
            print(f"총 거래 횟수: {total_trades:,}회")
            print(f"최종 보유량: {self.holdings:,.2f} ADA")
            
            if self.trades:
                df = pd.DataFrame(self.trades)
                df.to_csv(f'trade_history_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)
                print("\n거래 기록이 저장되었습니다.")

if __name__ == "__main__":
    live_test = ADALiveBacktest()
    live_test.run_live_test(duration_minutes=60)  # 1시간 테스트 