import requests
import time
from datetime import datetime
import hmac
import hashlib
import json
import jwt
import uuid
from urllib.parse import urlencode
import base64
import os
from dotenv import load_dotenv

class BithumbAPI:
    """
    빗썸 REST API 연동 클래스
    - Public API: 단순 GET 요청, 인증 불필요
    - Private API: JWT 인증 필요 (Authorization: Bearer ...)
    """

    def __init__(self, access_key=None, secret_key=None):
        # .env 파일 로드
        load_dotenv()
        
        self.base_url = "https://api.bithumb.com"
        # 환경 변수에서 API 키를 가져오거나, 직접 제공된 키를 사용
        self.access_key = access_key or os.getenv('BITHUMB_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('BITHUMB_SECRET_KEY')
        
        if not self.access_key or not self.secret_key:
            raise ValueError("API 키가 설정되지 않았습니다. .env 파일이나 직접 키를 제공해주세요.")

    # ---------------------------
    # Public API
    # ---------------------------
    def get_market_all(self, isDetails=True):
        """
        빗썸에서 거래 가능한 마켓과 가상자산 정보 조회
        """
        url = f"{self.base_url}/v1/market/all"
        params = {"isDetails": str(isDetails).lower()}
        headers = {"accept": "application/json"}
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def get_virtual_asset_warning(self):
        """
        경보제 API 호출
        """
        url = f"{self.base_url}/v1/market/virtual_asset_warning"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_candles_minutes(self, markets=None, unit=1, count=1, to=None):
        """
        여러 마켓의 분봉 데이터를 조회
        (참고: 일, 주, 월 기준 캔들 데이터 조회는 별도의 메서드로 구현 필요)
        
        Args:
            markets (list): 마켓 코드 리스트 (예: ["KRW-BTC", "KRW-ETH"])
                - None일 경우 ["KRW-BTC"]가 기본값으로 사용됨
            unit (int): 분 단위
                - 가능한 값: 1, 3, 5, 10, 15, 30, 60, 240
                - 기본값: 1
            count (int): 캔들 개수
                - 최대 200개까지 요청 가능
                - 기본값: 1
            to (str): 마지막 캔들 시각
                - ISO8601 포맷 (예: "2024-04-28T14:55:00")
                - None일 경우 가장 최근 데이터부터 조회
            
        Returns:
            dict: {마켓코드: 캔들데이터} 형태의 딕셔너리
                - 각 마켓의 캔들 데이터는 리스트 형태
                - 조회 실패 시 해당 마켓의 값은 None
        """
        if markets is None:
            markets = ["KRW-BTC"]  # 기본값
        
        results = {}
        for market in markets:
            try:
                url = f"{self.base_url}/v1/candles/minutes/{unit}"
                params = {
                    "market": market,
                    "count": count
                }
                if to:
                    params["to"] = to
                    
                headers = {"accept": "application/json"}
                response = requests.get(url, params=params, headers=headers)
                response.raise_for_status()
                results[market] = response.json()
            except Exception as e:
                print(f"{market} 캔들 데이터 조회 실패: {str(e)}")
                results[market] = None
            
        return results

    def get_trades_ticks(self, market, count=1, to=None, cursor=None, days_ago=None):
        """
        최근 체결 내역을 조회하는 메서드
        
        Args:
            market (str): 마켓 코드 (예: "KRW-BTC")
            count (int): 조회할 체결 개수 (기본값: 1)
            to (str): 마지막 체결 시각 (형식: HHmmss 또는 HH:mm:ss)
            cursor (str): 페이지네이션 커서 (sequentialId)
            days_ago (int): 최근 체결 날짜 기준 이전 데이터 조회 (1~7일)
            
        Returns:
            dict: 체결 내역 데이터
                - sequential_id: 체결의 유일성을 판단하기 위한 ID (체결 순서 보장하지 않음)
                - market: 마켓 코드
                - trade_date_utc: 체결 일자 (UTC)
                - trade_time_utc: 체결 시각 (UTC)
                - trade_price: 체결 가격
                - trade_volume: 체결량
                - trade_funds: 체결 금액
                - side: 체결 종류 (bid: 매수, ask: 매도)
                
        Raises:
            requests.exceptions.RequestException: API 요청 실패 시 발생
        """
        url = f"{self.base_url}/v1/trades/ticks"
        params = {
            "market": market,
            "count": count
        }
        
        # 선택적 파라미터 추가
        if to:
            params["to"] = to
        if cursor:
            params["cursor"] = cursor
        if days_ago:
            if not 1 <= days_ago <= 7:
                raise ValueError("days_ago는 1에서 7 사이의 값이어야 합니다.")
            params["daysAgo"] = days_ago
            
        headers = {"accept": "application/json"}
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"체결 내역 조회 실패: {str(e)}")
            raise

    def get_ticker(self, markets):
        """
        현재가 정보를 조회하는 메서드
        
        Args:
            markets (str): 반점으로 구분되는 마켓 코드 (예: "KRW-BTC" 또는 "KRW-BTC,BTC-ETH")
            
        Returns:
            dict: 현재가 정보
                - market: 종목 구분 코드
                - trade_date: 최근 거래 일자(UTC) yyyyMMdd
                - trade_time: 최근 거래 시각(UTC) HHmmss
                - trade_date_kst: 최근 거래 일자(KST) yyyyMMdd
                - trade_time_kst: 최근 거래 시각(KST) HHmmss
                - trade_timestamp: 최근 거래 일시(UTC) Unix Timestamp
                - opening_price: 시가
                - high_price: 고가
                - low_price: 저가
                - trade_price: 종가(현재가)
                - prev_closing_price: 전일 종가(KST 0시 기준)
                - change: EVEN(보합), RISE(상승), FALL(하락)
                - change_price: 변화액의 절대값
                - change_rate: 변화율의 절대값
                - signed_change_price: 부호가 있는 변화액
                - signed_change_rate: 부호가 있는 변화율
                - trade_volume: 가장 최근 거래량
                - acc_trade_price: 누적 거래대금(KST 0시 기준)
                - acc_trade_price_24h: 24시간 누적 거래대금
                - acc_trade_volume: 누적 거래량(KST 0시 기준)
                - acc_trade_volume_24h: 24시간 누적 거래량
                - highest_52_week_price: 52주 신고가
                - highest_52_week_date: 52주 신고가 달성일 yyyy-MM-dd
                - lowest_52_week_price: 52주 신저가
                - lowest_52_week_date: 52주 신저가 달성일 yyyy-MM-dd
                - timestamp: 타임스탬프
                
        Raises:
            requests.exceptions.RequestException: API 요청 실패 시 발생
        """
        url = f"{self.base_url}/v1/ticker"
        params = {"markets": markets}
        headers = {"accept": "application/json"}
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"현재가 정보 조회 실패: {str(e)}")
            raise

    def get_orderbook(self, markets):
        """
        호가 정보를 조회하는 메서드
        
        Args:
            markets (str): 반점으로 구분되는 마켓 코드 (예: "KRW-BTC" 또는 "KRW-BTC,BTC-ETH")
            
        Returns:
            dict: 호가 정보
                - market: 마켓 코드
                - timestamp: 호가 생성 시각
                - total_ask_size: 호가 매도 총 잔량
                - total_bid_size: 호가 매수 총 잔량
                - orderbook_units: 호가 정보 리스트
                    - ask_price: 매도호가
                    - bid_price: 매수호가
                    - ask_size: 매도 잔량
                    - bid_size: 매수 잔량
                
        주의사항:
            - 단일 마켓 코드 입력 시 30호가까지 정보 제공
            - 여러 마켓 코드 입력 시 15호가까지 정보 제공
            
        Raises:
            requests.exceptions.RequestException: API 요청 실패 시 발생
        """
        url = f"{self.base_url}/v1/orderbook"
        params = {"markets": markets}
        headers = {"accept": "application/json"}
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"호가 정보 조회 실패: {str(e)}")
            raise

    # ---------------------------
    # Private API
    # ---------------------------
    def _create_jwt_token(self, query_params=None):
        """
        JWT 인증 토큰을 생성하는 내부 메서드
        
        Args:
            query_params (dict, optional): 쿼리 파라미터 (파라미터가 있는 경우 필수)
            
        Returns:
            str: JWT 인증 토큰
            
        Raises:
            ValueError: API Key나 Secret Key가 설정되지 않은 경우
            requests.exceptions.RequestException: API 요청 실패 시 발생
            
        주의사항:
            - 파라미터가 있는 경우 query_hash를 생성해야 합니다.
            - 배열 파라미터는 key[]=value1&key[]=value2 형식으로 인코딩됩니다.
            - JWT 토큰은 30초 이내에 사용해야 합니다.
            - IP 제한이 설정된 경우 허용된 IP에서만 API를 사용할 수 있습니다.
        """
        if not self.access_key or not self.secret_key:
            raise ValueError("API Key와 Secret Key가 필요합니다.")
            
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': round(time.time() * 1000)
        }
        
        # 파라미터가 있는 경우 query_hash 생성
        if query_params:
            # 배열 파라미터 처리
            encoded_params = urlencode(query_params, doseq=True).encode()
            hash_obj = hashlib.sha512()
            hash_obj.update(encoded_params)
            payload['query_hash'] = hash_obj.hexdigest()
            payload['query_hash_alg'] = 'SHA512'
            
        try:
            jwt_token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            return f'Bearer {jwt_token}'
        except Exception as e:
            error_msg = "JWT 토큰 생성 실패: "
            if "expired" in str(e):
                error_msg += "토큰이 만료되었습니다. (expired_jwt)"
            elif "verification" in str(e):
                error_msg += "토큰 검증에 실패했습니다. (jwt_verification)"
            elif "query" in str(e):
                error_msg += "쿼리 검증에 실패했습니다. (invalid_query_payload)"
            else:
                error_msg += str(e)
            raise ValueError(error_msg)

    def get_api_keys(self):
        """
        API 키 리스트와 만료 일자를 조회하는 메서드
        
        Returns:
            list: API 키 정보 리스트
                - access_key: API KEY
                - expire_at: 만료일시
                
        Raises:
            ValueError: API Key나 Secret Key가 설정되지 않은 경우
            requests.exceptions.RequestException: API 요청 실패 시 발생
            
        주의사항:
            - API Key는 발급일 기준으로 1년 동안 사용 가능
            - 기간 연장은 불가능하며, 1년 경과 시 재발급 필요
        """
        url = f"{self.base_url}/v1/api_keys"
        headers = {
            'Authorization': self._create_jwt_token(),
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API 키 리스트 조회 실패: {str(e)}")
            raise

    def get_accounts(self):
        """
        전체 계좌 정보를 조회하는 메서드
        
        Returns:
            list: 계좌 정보 리스트
                - currency: 화폐 코드 (예: "BTC", "ETH")
                - balance: 주문가능 금액/수량
                - locked: 주문 중 묶여있는 금액/수량
                - avg_buy_price: 매수평균가
                - avg_buy_price_modified: 매수평균가 수정 여부
                - unit_currency: 평단가 기준 화폐
                
        Raises:
            ValueError: API Key나 Secret Key가 설정되지 않은 경우
            requests.exceptions.RequestException: API 요청 실패 시 발생
        """
        url = f"{self.base_url}/v1/accounts"
        headers = {
            'Authorization': self._create_jwt_token(),
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"계좌 정보 조회 실패: {str(e)}")
            raise

    def get_order_chance(self, market):
        """
        주문 가능 정보를 조회하는 메서드
        
        Args:
            market (str): 마켓 ID (예: "KRW-BTC")
            
        Returns:
            dict: 주문 가능 정보
                - bid_fee: 매수 수수료 비율
                - ask_fee: 매도 수수료 비율
                - maker_bid_fee: 마켓 매수 수수료 비율
                - maker_ask_fee: 마켓 매도 수수료 비율
                - market: 마켓 정보
                    - id: 마켓의 유일 키
                    - name: 마켓 이름
                    - order_types: 지원 주문 방식
                    - ask_types: 매도 주문 지원 방식
                    - bid_types: 매수 주문 지원 방식
                    - order_sides: 지원 주문 종류
                    - bid: 매수 시 제약사항
                        - currency: 화폐 코드
                        - price_unit: 주문금액 단위
                        - min_total: 최소 매수 금액
                    - ask: 매도 시 제약사항
                        - currency: 화폐 코드
                        - price_unit: 주문금액 단위
                        - min_total: 최소 매도 금액
                    - max_total: 최대 매도/매수 금액
                    - state: 마켓 운영 상태
                - bid_account: 매수 계좌 상태
                    - currency: 화폐 코드
                    - balance: 주문가능 금액/수량
                    - locked: 주문 중 묶여있는 금액/수량
                    - avg_buy_price: 매수평균가
                    - avg_buy_price_modified: 매수평균가 수정 여부
                    - unit_currency: 평단가 기준 화폐
                - ask_account: 매도 계좌 상태
                    - currency: 화폐 코드
                    - balance: 주문가능 금액/수량
                    - locked: 주문 중 묶여있는 금액/수량
                    - avg_buy_price: 매수평균가
                    - avg_buy_price_modified: 매수평균가 수정 여부
                    - unit_currency: 평단가 기준 화폐
                    
        Raises:
            ValueError: API Key나 Secret Key가 설정되지 않은 경우
            requests.exceptions.RequestException: API 요청 실패 시 발생
        """
        url = f"{self.base_url}/v1/orders/chance"
        params = {"market": market}
        
        # JWT 토큰 생성
        headers = {
            'Authorization': self._create_jwt_token(params),
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"주문 가능 정보 조회 실패: {str(e)}")
            raise

    def get_order(self, uuid: str) -> dict:
        """
        주문 UUID로 해당 주문의 내역을 조회하는 메서드
        
        Args:
            uuid (str): 주문 UUID
            
        Returns:
            dict: 주문 정보
                - uuid: 주문의 고유 아이디
                - side: 주문 종류 (bid: 매수, ask: 매도)
                - ord_type: 주문 방식 (limit: 지정가)
                - price: 주문 당시 화폐 가격
                - state: 주문 상태
                - market: 마켓의 유일키
                - created_at: 주문 생성 시간
                - volume: 사용자가 입력한 주문 양
                - remaining_volume: 체결 후 남은 주문 양
                - reserved_fee: 수수료로 예약된 비용
                - remaining_fee: 남은 수수료
                - paid_fee: 사용된 수수료
                - locked: 거래에 사용중인 비용
                - executed_volume: 체결된 양
                - trades_count: 해당 주문에 걸린 체결 수
                - trades: 체결 내역 리스트
                    - market: 마켓의 유일 키
                    - uuid: 체결의 고유 아이디
                    - price: 체결 가격
                    - volume: 체결 양
                    - funds: 체결된 총 가격
                    - side: 체결 종류
                    - created_at: 체결 시각
            
        Raises:
            ValueError: API Key나 Secret Key가 설정되지 않은 경우
            requests.exceptions.RequestException: API 요청 실패 시 발생
        """
        url = f"{self.base_url}/v1/order"
        params = {"uuid": uuid}
        
        # JWT 토큰 생성
        headers = {
            'Authorization': self._create_jwt_token(params),
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"주문 조회 실패: {str(e)}")
            raise

    def get_orders(self, market: str, state: str = None, states: list = None, 
                  uuids: list = None, page: int = 1, limit: int = 100, 
                  order_by: str = 'desc') -> dict:
        """
        주문 리스트를 조회하는 메서드
        
        Args:
            market (str): 마켓 ID (예: "KRW-BTC")
            state (str, optional): 주문 상태 (wait, watch, done, cancel)
            states (list, optional): 주문 상태 목록
            uuids (list, optional): 주문 UUID 목록
            page (int, optional): 페이지 번호 (기본값: 1)
            limit (int, optional): 조회 개수 제한 (기본값: 100)
            order_by (str, optional): 정렬 방식 (asc, desc, 기본값: desc)
            
        Returns:
            dict: 주문 리스트 정보
                - uuid: 주문의 고유 아이디
                - side: 주문 종류 (bid: 매수, ask: 매도)
                - ord_type: 주문 방식 (limit: 지정가)
                - price: 주문 당시 화폐 가격
                - state: 주문 상태
                - market: 마켓의 유일키
                - created_at: 주문 생성 시간
                - volume: 주문 수량
                - remaining_volume: 남은 주문 수량
                - reserved_fee: 예약된 수수료
                - remaining_fee: 남은 수수료
                - paid_fee: 사용된 수수료
                - locked: 거래 중인 금액
                - executed_volume: 체결된 수량
                - trades_count: 체결 건수
                
        Raises:
            ValueError: API Key나 Secret Key가 설정되지 않은 경우
            requests.exceptions.RequestException: API 요청 실패 시 발생
        """
        url = f"{self.base_url}/v1/orders"
        
        # 파라미터 구성
        params = {
            'market': market,
            'page': page,
            'limit': limit,
            'order_by': order_by
        }
        
        # 선택적 파라미터 추가
        if state:
            params['state'] = state
        if states:
            params['states'] = states
        if uuids:
            params['uuids'] = uuids
            
        # JWT 토큰 생성
        headers = {
            'Authorization': self._create_jwt_token(params),
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"주문 리스트 조회 실패: {str(e)}")
            raise

    def cancel_order(self, uuid: str) -> dict:
        """
        주문을 취소하는 메서드
        
        Args:
            uuid (str): 취소할 주문의 UUID
            
        Returns:
            dict: 취소된 주문 정보
                - uuid: 주문의 고유 아이디
                - side: 주문 종류 (bid: 매수, ask: 매도)
                - ord_type: 주문 방식 (limit: 지정가)
                - price: 주문 당시 화폐 가격
                - state: 주문 상태
                - market: 마켓의 유일키
                - created_at: 주문 생성 시간
                - volume: 주문 수량
                - remaining_volume: 남은 주문 수량
                - reserved_fee: 예약된 수수료
                - remaining_fee: 남은 수수료
                - paid_fee: 사용된 수수료
                - locked: 거래 중인 금액
                - executed_volume: 체결된 수량
                - trades_count: 체결 건수
                
        Raises:
            ValueError: API Key나 Secret Key가 설정되지 않은 경우
            requests.exceptions.RequestException: API 요청 실패 시 발생
            
        주의사항:
            - 취소하려는 주문이 존재하지 않거나 이미 취소된 경우 에러가 발생할 수 있습니다.
            - 주문 취소는 즉시 처리되지 않을 수 있으며, 상태가 'cancel'로 변경되는 데 시간이 걸릴 수 있습니다.
        """
        url = f"{self.base_url}/v1/order"
        
        # 파라미터 구성
        params = {'uuid': uuid}
            
        # JWT 토큰 생성
        headers = {
            'Authorization': self._create_jwt_token(params),
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        try:
            response = requests.delete(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"주문 취소 실패: {str(e)}")
            raise

    def create_order(self, market: str, side: str, volume: str, price: str, ord_type: str) -> dict:
        """
        주문을 생성하는 메서드
        
        Args:
            market (str): 마켓 ID (예: "KRW-BTC")
            side (str): 주문 종류 (bid: 매수, ask: 매도)
            volume (str): 주문 수량
            price (str): 주문 가격
            ord_type (str): 주문 타입 (limit: 지정가, price: 시장가 매수, market: 시장가 매도)
            
        Returns:
            dict: 생성된 주문 정보
                - uuid: 주문의 고유 아이디
                - side: 주문 종류 (bid: 매수, ask: 매도)
                - ord_type: 주문 방식 (limit: 지정가)
                - price: 주문 당시 화폐 가격
                - state: 주문 상태
                - market: 마켓의 유일키
                - created_at: 주문 생성 시간
                - volume: 주문 수량
                - remaining_volume: 남은 주문 수량
                - reserved_fee: 예약된 수수료
                - remaining_fee: 남은 수수료
                - paid_fee: 사용된 수수료
                - locked: 거래 중인 금액
                - executed_volume: 체결된 수량
                - trades_count: 체결 건수
                
        Raises:
            ValueError: API Key나 Secret Key가 설정되지 않은 경우
            requests.exceptions.RequestException: API 요청 실패 시 발생
            
        주의사항:
            - 지정가 주문 시 본인의 미체결 주문과 체결될 확률이 높은 주문은 주문이 불가능할 수 있습니다.
            - 시장가 주문 시 주문 가격과 수량 설정에 주의가 필요합니다.
            - 주문 생성 후 상태가 'wait'로 변경되는 데 시간이 걸릴 수 있습니다.
        """
        url = f"{self.base_url}/v1/orders"
        
        # 요청 바디 구성
        request_body = {
            'market': market,
            'side': side,
            'volume': volume,
            'price': price,
            'ord_type': ord_type
        }
            
        # JWT 토큰 생성
        headers = {
            'Authorization': self._create_jwt_token(request_body),
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(url, data=json.dumps(request_body), headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"주문 생성 실패: {str(e)}")
            raise

