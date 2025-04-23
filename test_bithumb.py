import asyncio
from bithumb_api import BithumbAPI

async def test_public_api():
    """공개 API 테스트"""
    api = BithumbAPI()
    
    print("\n=== 공개 API 테스트 ===")
    
    # 1. 마켓 정보 조회
    print("\n1. 전체 마켓 정보:")
    markets = await api.get_market_all()
    print(markets)
    
    # 2. BTC 1분봉 데이터
    print("\n2. BTC 1분봉 데이터:")
    candles = await api.get_candles_minutes("KRW-BTC", count=1)
    print(candles)
    
    # 3. 현재가 정보
    print("\n3. BTC 현재가:")
    ticker = await api.get_ticker("KRW-BTC")
    print(ticker)

async def test_private_api():
    """개인 API 테스트"""
    api = BithumbAPI()
    
    print("\n=== 개인 API 테스트 ===")
    
    # 1. 잔고 조회
    print("\n1. 계좌 잔고:")
    balance = await api.get_balance()
    print(balance)
    
    # 2. 주문하기 (실제 주문은 주석 처리)
    """
    print("\n2. 주문 테스트:")
    order = await api.place_order(
        symbol="BTC",
        side="bid",  # 매수
        volume=0.001,
        price=50000000
    )
    print(order)
    """

# 테스트 실행
if __name__ == "__main__":
    # 공개 API 테스트
    print("공개 API 테스트를 시작합니다...")
    asyncio.run(test_public_api())
    
    # 개인 API 테스트
    response = input("\n개인 API 테스트를 진행하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(test_private_api()) 