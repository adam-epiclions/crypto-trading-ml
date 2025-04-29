from bithumb_api import BithumbAPI
import json

def test_market_all():
    api = BithumbAPI()
    try:
        result = api.get_market_all()
        print("\n=== 마켓 전체 정보 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"마켓 정보 조회 실패: {str(e)}")
        return False

def translate_warning_type(warning_type):
    """경보 유형을 한글로 변환"""
    warning_types = {
        "PRICE_SUDDEN_FLUCTUATION": "가격 급등락",
        "TRADING_VOLUME_SUDDEN_FLUCTUATION": "거래량 급등",
        "DEPOSIT_AMOUNT_SUDDEN_FLUCTUATION": "입금량 급등",
        "PRICE_DIFFERENCE_HIGH": "가격 차이",
        "SPECIFIC_ACCOUNT_HIGH_TRANSACTION": "소수계좌 거래 집중",
        "EXCHANGE_TRADING_CONCENTRATION": "거래소 거래 집중"
    }
    return warning_types.get(warning_type, warning_type)

def test_virtual_asset_warning():
    api = BithumbAPI()
    try:
        result = api.get_virtual_asset_warning()
        print("\n=== 경보제 정보 ===")
        
        # 경보 유형을 한글로 변환
        if isinstance(result, list):
            for item in result:
                if "warning_type" in item:
                    item["warning_type"] = translate_warning_type(item["warning_type"])
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"경보제 정보 조회 실패: {str(e)}")
        return False

def test_minute_candles():
    api = BithumbAPI()
    try:
        # 여러 마켓의 5분봉 데이터 조회 (최근 5개)
        markets = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
        result = api.get_candles_minutes(markets=markets, unit=5, count=5)
        print("\n=== 5분봉 캔들 데이터 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"분 캔들 데이터 조회 실패: {str(e)}")
        return False

def test_trades_ticks():
    """
    체결 내역 조회 테스트
    다양한 파라미터 조합으로 테스트를 수행합니다.
    - 최대 7일 이내의 데이터만 조회 가능
    """
    api = BithumbAPI()
    test_cases = [
        {
            "name": "기본 조회 (최근 1개 체결)",
            "params": {"market": "KRW-BTC", "count": 1}
        },
        {
            "name": "여러 개의 체결 조회",
            "params": {"market": "KRW-BTC", "count": 5}
        },
        {
            "name": "특정 시간 이후 체결 조회",
            "params": {"market": "KRW-BTC", "to": "143000", "count": 3}
        },
        {
            "name": "3일 전 체결 조회",
            "params": {"market": "KRW-BTC", "days_ago": 3, "count": 3}
        },
        {
            "name": "7일 전 체결 조회",
            "params": {"market": "KRW-BTC", "days_ago": 7, "count": 3}
        }
    ]
    
    print("\n=== 체결 내역 조회 테스트 ===")
    print("주의: 체결 내역은 최대 7일 이내의 데이터만 조회 가능합니다.")
    
    for case in test_cases:
        try:
            print(f"\n테스트 케이스: {case['name']}")
            result = api.get_trades_ticks(**case['params'])
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"체결 내역 조회 실패: {str(e)}")
            return False
    
    return True

def test_ticker():
    """
    현재가 정보 조회 테스트
    단일 마켓과 여러 마켓에 대한 테스트를 수행합니다.
    """
    api = BithumbAPI()
    test_cases = [
        {
            "name": "단일 마켓 조회",
            "params": {"markets": "KRW-BTC"}
        },
        {
            "name": "여러 마켓 조회",
            "params": {"markets": "KRW-BTC,KRW-ETH,KRW-XRP"}
        }
    ]
    
    print("\n=== 현재가 정보 조회 테스트 ===")
    
    for case in test_cases:
        try:
            print(f"\n테스트 케이스: {case['name']}")
            result = api.get_ticker(**case['params'])
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"현재가 정보 조회 실패: {str(e)}")
            return False
    
    return True

def test_orderbook():
    """
    호가 정보 조회 테스트
    단일 마켓과 여러 마켓에 대한 테스트를 수행합니다.
    """
    api = BithumbAPI()
    test_cases = [
        {
            "name": "단일 마켓 조회 (30호가)",
            "params": {"markets": "KRW-BTC"}
        },
        {
            "name": "여러 마켓 조회 (15호가)",
            "params": {"markets": "KRW-BTC,KRW-ETH,KRW-XRP"}
        }
    ]
    
    print("\n=== 호가 정보 조회 테스트 ===")
    print("주의: 단일 마켓은 30호가, 여러 마켓은 15호가까지 정보를 제공합니다.")
    
    for case in test_cases:
        try:
            print(f"\n테스트 케이스: {case['name']}")
            result = api.get_orderbook(**case['params'])
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"호가 정보 조회 실패: {str(e)}")
            return False
    
    return True

if __name__ == "__main__":
    print("마켓 정보 테스트 시작...")
    market_success = test_market_all()
    
    print("\n경보제 정보 테스트 시작...")
    warning_success = test_virtual_asset_warning()
    
    print("\n분 캔들 데이터 테스트 시작...")
    candles_success = test_minute_candles()
    
    print("\n체결 내역 조회 테스트 시작...")
    trades_success = test_trades_ticks()
    
    print("\n현재가 정보 조회 테스트 시작...")
    ticker_success = test_ticker()
    
    print("\n호가 정보 조회 테스트 시작...")
    orderbook_success = test_orderbook()
    
    print("\n=== 테스트 결과 ===")
    print(f"마켓 정보 조회: {'성공' if market_success else '실패'}")
    print(f"경보제 정보 조회: {'성공' if warning_success else '실패'}")
    print(f"분 캔들 데이터 조회: {'성공' if candles_success else '실패'}")
    print(f"체결 내역 조회: {'성공' if trades_success else '실패'}")
    print(f"현재가 정보 조회: {'성공' if ticker_success else '실패'}")
    print(f"호가 정보 조회: {'성공' if orderbook_success else '실패'}") 