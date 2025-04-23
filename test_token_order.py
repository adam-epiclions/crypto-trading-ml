import asyncio
from bithumb_api import BithumbAPI

async def test_token_order(token="ADA", volume=10):
    """
    토큰 주문 테스트 함수
    :param token: 거래할 토큰 심볼 (예: "ADA", "BTC", "XRP" 등)
    :param volume: 주문할 수량
    """
    api = BithumbAPI()
    
    try:
        # 1. 현재 토큰 시장 가격 확인
        ticker = await api.get_ticker(f"KRW-{token}")
        current_price = float(ticker[0]['trade_price'])
        
        print(f"현재 {token} 가격: {current_price}원")
        
        # 2. 테스트 주문
        test_price = int(current_price * 0.99)  # 현재가의 99%
        
        print(f"\n주문 예정:")
        print(f"토큰: {token}")
        print(f"가격: {test_price}원")
        print(f"수량: {volume} {token}")
        print(f"총 주문금액: {test_price * volume}원")
        
        confirm = input("\n이대로 주문하시겠습니까? (y/n): ")
        
        if confirm.lower() == 'y':
            order = await api.place_order(
                symbol=token,
                side="bid",
                volume=volume,
                price=test_price,
                ord_type='limit'
            )
            print("\n주문 결과:", order)
        else:
            print("\n주문이 취소되었습니다.")
            
    except Exception as e:
        print("에러 발생:", str(e))

if __name__ == "__main__":
    # 토큰 심볼과 수량을 입력받음
    token = input("거래할 토큰 심볼을 입력하세요 (예: ADA, BTC, XRP): ").upper()
    volume = float(input(f"{token} 주문 수량을 입력하세요: "))
    
    asyncio.run(test_token_order(token, volume)) 