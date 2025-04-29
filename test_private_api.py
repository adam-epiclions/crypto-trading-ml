from bithumb_api import BithumbAPI
import json
import os
from dotenv import load_dotenv
import requests

def test_accounts():
    """
    계좌 정보 조회 테스트
    API Key와 Secret Key가 설정된 경우에만 테스트를 수행합니다.
    """
    # .env 파일에서 환경 변수 로드
    load_dotenv()
    
    # 환경 변수에서 API 키 가져오기
    access_key = os.getenv('BITHUMB_ACCESS_KEY')
    secret_key = os.getenv('BITHUMB_SECRET_KEY')
    
    if not access_key or not secret_key:
        print("환경 변수에서 API 키를 찾을 수 없습니다.")
        print("다음과 같이 .env 파일을 생성해주세요:")
        print("BITHUMB_ACCESS_KEY=your_access_key")
        print("BITHUMB_SECRET_KEY=your_secret_key")
        return False
    
    api = BithumbAPI(access_key=access_key, secret_key=secret_key)
    
    print("\n=== 계좌 정보 조회 테스트 ===")
    print("주의: API Key와 Secret Key가 설정되어 있어야 합니다.")
    
    try:
        result = api.get_accounts()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"계좌 정보 조회 실패: {str(e)}")
        return False

def test_order_chance():
    """주문 가능 정보 조회 테스트"""
    try:
        # API 키가 있는지 확인
        if not os.getenv('BITHUMB_ACCESS_KEY') or not os.getenv('BITHUMB_SECRET_KEY'):
            print("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            return
            
        # BithumbAPI 인스턴스 생성
        api = BithumbAPI(
            access_key=os.getenv('BITHUMB_ACCESS_KEY'),
            secret_key=os.getenv('BITHUMB_SECRET_KEY')
        )
        
        # 주문 가능 정보 조회
        market = "KRW-BTC"  # 비트코인 마켓
        result = api.get_order_chance(market)
        
        # API 응답 전체 출력
        print("\n=== API 응답 전체 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # market.bid 객체 확인
        print("\n=== market.bid 객체 ===")
        print(json.dumps(result['market']['bid'], indent=2, ensure_ascii=False))
        
        # 결과 출력
        print("\n=== 주문 가능 정보 ===")
        print(f"마켓: {market}")
        print(f"매수 수수료: {result['bid_fee']}")
        print(f"매도 수수료: {result['ask_fee']}")
        print(f"마켓 매수 수수료: {result['maker_bid_fee']}")
        print(f"마켓 매도 수수료: {result['maker_ask_fee']}")
        
        # 마켓 정보
        market_info = result['market']
        print("\n=== 마켓 정보 ===")
        print(f"마켓 ID: {market_info['id']}")
        print(f"마켓 이름: {market_info['name']}")
        print(f"지원 주문 방식: {market_info['order_types']}")
        print(f"매도 주문 지원 방식: {market_info['ask_types']}")
        print(f"매수 주문 지원 방식: {market_info['bid_types']}")
        print(f"지원 주문 종류: {market_info['order_sides']}")
        
        # 매수 제약사항
        bid = market_info['bid']
        print("\n=== 매수 제약사항 ===")
        print(f"화폐: {bid['currency']}")
        if 'price_unit' in bid:
            print(f"주문금액 단위: {bid['price_unit']}")
        print(f"최소 매수 금액: {bid['min_total']}")
        
        # 매도 제약사항
        ask = market_info['ask']
        print("\n=== 매도 제약사항 ===")
        print(f"화폐: {ask['currency']}")
        if 'price_unit' in ask:
            print(f"주문금액 단위: {ask['price_unit']}")
        print(f"최소 매도 금액: {ask['min_total']}")
        
        # 계좌 정보
        bid_account = result['bid_account']
        print("\n=== 매수 계좌 정보 ===")
        print(f"화폐: {bid_account['currency']}")
        print(f"주문가능 금액: {bid_account['balance']}")
        print(f"주문 중 금액: {bid_account['locked']}")
        print(f"매수평균가: {bid_account['avg_buy_price']}")
        
        ask_account = result['ask_account']
        print("\n=== 매도 계좌 정보 ===")
        print(f"화폐: {ask_account['currency']}")
        print(f"주문가능 수량: {ask_account['balance']}")
        print(f"주문 중 수량: {ask_account['locked']}")
        print(f"매수평균가: {ask_account['avg_buy_price']}")
        
        return True
        
    except Exception as e:
        print(f"주문 가능 정보 조회 중 오류 발생: {str(e)}")
        return False

def test_orders():
    """주문 리스트 조회 테스트"""
    try:
        # API 키가 있는지 확인
        if not os.getenv('BITHUMB_ACCESS_KEY') or not os.getenv('BITHUMB_SECRET_KEY'):
            print("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            return
            
        # BithumbAPI 인스턴스 생성
        api = BithumbAPI(
            access_key=os.getenv('BITHUMB_ACCESS_KEY'),
            secret_key=os.getenv('BITHUMB_SECRET_KEY')
        )
        
        # 주문 리스트 조회
        market = "KRW-BTC"  # 비트코인 마켓
        result = api.get_orders(market, state='done', page=1, limit=10)
        
        # API 응답 전체 출력
        print("\n=== API 응답 전체 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 주문 리스트 출력
        if result:
            print("\n=== 주문 리스트 ===")
            for order in result:
                print(f"\n주문 UUID: {order['uuid']}")
                print(f"주문 종류: {order['side']}")  # bid: 매수, ask: 매도
                print(f"주문 방식: {order['ord_type']}")  # limit: 지정가
                print(f"주문 가격: {order['price']}")
                print(f"주문 상태: {order['state']}")
                print(f"마켓: {order['market']}")
                print(f"주문 생성 시간: {order['created_at']}")
                print(f"주문 수량: {order['volume']}")
                print(f"남은 수량: {order['remaining_volume']}")
                print(f"체결된 수량: {order['executed_volume']}")
                print(f"체결 건수: {order['trades_count']}")
                print("---")
        
        return True
        
    except Exception as e:
        print(f"주문 리스트 조회 중 오류 발생: {str(e)}")
        return False

def test_get_order():
    """개별 주문 조회 테스트"""
    try:
        # API 키가 있는지 확인
        if not os.getenv('BITHUMB_ACCESS_KEY') or not os.getenv('BITHUMB_SECRET_KEY'):
            print("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            return
            
        # BithumbAPI 인스턴스 생성
        api = BithumbAPI(
            access_key=os.getenv('BITHUMB_ACCESS_KEY'),
            secret_key=os.getenv('BITHUMB_SECRET_KEY')
        )
        
        # 주문 UUID 입력 받기
        uuid = input("조회할 주문 UUID를 입력하세요: ")
        
        # 주문 정보 조회
        result = api.get_order(uuid)
        
        # API 응답 전체 출력
        print("\n=== API 응답 전체 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 주문 정보 출력
        print("\n=== 주문 정보 ===")
        print(f"주문 UUID: {result['uuid']}")
        print(f"주문 종류: {result['side']}")  # bid: 매수, ask: 매도
        print(f"주문 방식: {result['ord_type']}")  # limit: 지정가
        print(f"주문 가격: {result['price']}")
        print(f"주문 상태: {result['state']}")
        print(f"마켓: {result['market']}")
        print(f"주문 생성 시간: {result['created_at']}")
        if 'volume' in result:
            print(f"주문 수량: {result['volume']}")
        if 'remaining_volume' in result:
            print(f"남은 수량: {result['remaining_volume']}")
        if 'executed_volume' in result:
            print(f"체결된 수량: {result['executed_volume']}")
        if 'trades_count' in result:
            print(f"체결 건수: {result['trades_count']}")
        
        # 체결 내역 출력
        if result['trades']:
            print("\n=== 체결 내역 ===")
            for trade in result['trades']:
                print(f"체결 UUID: {trade['uuid']}")
                print(f"체결 가격: {trade['price']}")
                print(f"체결 수량: {trade['volume']}")
                print(f"체결 금액: {trade['funds']}")
                print(f"체결 종류: {trade['side']}")  # bid: 매수, ask: 매도
                print(f"체결 시간: {trade['created_at']}")
                print("---")
        
        return True
        
    except Exception as e:
        print(f"주문 조회 중 오류 발생: {str(e)}")
        return False

def test_create_order():
    """주문하기 테스트"""
    try:
        # API 키가 있는지 확인
        if not os.getenv('BITHUMB_ACCESS_KEY') or not os.getenv('BITHUMB_SECRET_KEY'):
            print("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            return
            
        # BithumbAPI 인스턴스 생성
        api = BithumbAPI(
            access_key=os.getenv('BITHUMB_ACCESS_KEY'),
            secret_key=os.getenv('BITHUMB_SECRET_KEY')
        )
        
        # 주문 정보 입력 받기
        market = input("마켓 ID를 입력하세요 (예: KRW-BTC): ")
        side = input("주문 종류를 입력하세요 (bid: 매수, ask: 매도): ")
        ord_type = input("주문 타입을 입력하세요 (limit: 지정가, price: 시장가 매수, market: 시장가 매도): ")
        
        # 주문 파라미터 설정
        if ord_type == 'price':  # 시장가 매수
            price = input("매수할 금액을 입력하세요 (KRW): ")
            # 시장가 매수는 volume을 빈 문자열로 전달
            result = api.create_order(market, side, "", price, ord_type)
        elif ord_type == 'market':  # 시장가 매도
            volume = input("매도할 수량을 입력하세요: ")
            # 시장가 매도는 price를 빈 문자열로 전달
            result = api.create_order(market, side, volume, "", ord_type)
        else:  # 지정가 주문
            volume = input("주문 수량을 입력하세요: ")
            price = input("주문 가격을 입력하세요: ")
            result = api.create_order(market, side, volume, price, ord_type)
        
        # API 응답 전체 출력
        print("\n=== API 응답 전체 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 생성된 주문 정보 출력
        print("\n=== 생성된 주문 정보 ===")
        print(f"주문 UUID: {result['uuid']}")
        print(f"주문 종류: {result['side']}")  # bid: 매수, ask: 매도
        print(f"주문 방식: {result['ord_type']}")  # limit: 지정가, price: 시장가 매수, market: 시장가 매도
        if 'price' in result:
            print(f"주문 가격: {result['price']}")
        print(f"주문 상태: {result['state']}")
        print(f"마켓: {result['market']}")
        print(f"주문 생성 시간: {result['created_at']}")
        if 'volume' in result:
            print(f"주문 수량: {result['volume']}")
        if 'remaining_volume' in result:
            print(f"남은 수량: {result['remaining_volume']}")
        if 'executed_volume' in result:
            print(f"체결된 수량: {result['executed_volume']}")
        if 'trades_count' in result:
            print(f"체결 건수: {result['trades_count']}")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            error_data = e.response.json()
            error_code = error_data.get('error', {}).get('code')
            error_message = error_data.get('error', {}).get('message', '')
            
            if error_code == 'invalid_query_payload':
                print(f"JWT 쿼리 검증 실패: {error_message}")
            elif error_code == 'jwt_verification':
                print(f"JWT 토큰 검증 실패: {error_message}")
            elif error_code == 'expired_jwt':
                print(f"JWT 토큰 만료: {error_message}")
            elif error_code == 'NotAllowIP':
                print(f"허용되지 않은 IP: {error_message}")
            elif error_code == 'out_of_scope':
                print(f"권한 부족: {error_message}")
            else:
                print(f"인증 실패: {error_message}")
        elif e.response.status_code == 400:
            error_data = e.response.json()
            error_message = error_data.get('error', {}).get('message', '잘못된 요청입니다.')
            print(f"잘못된 요청: {error_message}")
        elif e.response.status_code == 429:
            print("요청 횟수 초과: 잠시 후 다시 시도해주세요.")
        elif e.response.status_code >= 500:
            print("서버 오류: 잠시 후 다시 시도해주세요.")
        else:
            print(f"HTTP 오류 발생: {str(e)}")
        return False
    except Exception as e:
        print(f"주문 생성 중 오류 발생: {str(e)}")
        return False

def test_cancel_order():
    """주문 취소 테스트"""
    try:
        # API 키가 있는지 확인
        if not os.getenv('BITHUMB_ACCESS_KEY') or not os.getenv('BITHUMB_SECRET_KEY'):
            print("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            return
            
        # BithumbAPI 인스턴스 생성
        api = BithumbAPI(
            access_key=os.getenv('BITHUMB_ACCESS_KEY'),
            secret_key=os.getenv('BITHUMB_SECRET_KEY')
        )
        
        # 취소할 주문 UUID 입력 받기
        uuid = input("취소할 주문 UUID를 입력하세요: ")
        
        # 주문 취소
        result = api.cancel_order(uuid)
        
        # API 응답 전체 출력
        print("\n=== API 응답 전체 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 취소된 주문 정보 출력
        print("\n=== 취소된 주문 정보 ===")
        print(f"주문 UUID: {result['uuid']}")
        print(f"주문 종류: {result['side']}")  # bid: 매수, ask: 매도
        print(f"주문 방식: {result['ord_type']}")  # limit: 지정가
        print(f"주문 가격: {result['price']}")
        print(f"주문 상태: {result['state']}")
        print(f"마켓: {result['market']}")
        print(f"주문 생성 시간: {result['created_at']}")
        print(f"주문 수량: {result['volume']}")
        print(f"남은 수량: {result['remaining_volume']}")
        print(f"체결된 수량: {result['executed_volume']}")
        print(f"체결 건수: {result['trades_count']}")
        
        return True
        
    except Exception as e:
        print(f"주문 취소 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    print("Private API 테스트 시작...")
    
    print("\n계좌 정보 테스트 시작...")
    accounts_success = test_accounts()
    
    print("\n주문 가능 정보 테스트 시작...")
    order_chance_success = test_order_chance()
    
    print("\n주문 리스트 조회 테스트 시작...")
    orders_success = test_orders()
    
    print("\n개별 주문 조회 테스트 시작...")
    order_success = test_get_order()
    
    print("\n주문하기 테스트 시작...")
    create_success = test_create_order()
    
    print("\n주문 취소 테스트 시작...")
    cancel_success = test_cancel_order()
    
    print("\n=== 테스트 결과 ===")
    print(f"계좌 정보 조회: {'성공' if accounts_success else '실패'}")
    print(f"주문 가능 정보 조회: {'성공' if order_chance_success else '실패'}")
    print(f"주문 리스트 조회: {'성공' if orders_success else '실패'}")
    print(f"개별 주문 조회: {'성공' if order_success else '실패'}")
    print(f"주문하기: {'성공' if create_success else '실패'}")
    print(f"주문 취소: {'성공' if cancel_success else '실패'}") 