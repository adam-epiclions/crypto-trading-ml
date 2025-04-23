import requests
import time
from datetime import datetime
import hmac
import hashlib
import json
import jwt
import uuid
from urllib.parse import urlencode

class BithumbAPI:
    def __init__(self):
        self.base_url = "https://api.bithumb.com"
        self.connect_key = "key"
        self.secret_key = "secret"
        self.headers = {"accept": "application/json"}

    # 공개 API 메서드들
    async def get_market_all(self):
        """모든 마켓 정보 조회"""
        try:
            response = requests.get(f"{self.base_url}/v1/market/all?isDetails=false", 
                                  headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"마켓 정보 조회 실패: {str(e)}")

    async def get_candles_minutes(self, market, count=1, unit=1):
        """분봉 데이터 조회"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/candles/minutes/{unit}?market={market}&count={count}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"분봉 데이터 조회 실패: {str(e)}")

    async def get_candles_days(self, count=1):
        """일봉 데이터 조회"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/candles/days?count={count}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"일봉 데이터 조회 실패: {str(e)}")

    async def get_candles_weeks(self, count=1):
        """주봉 데이터 조회"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/candles/weeks?count={count}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"주봉 데이터 조회 실패: {str(e)}")

    async def get_recent_trades(self, market, count=1):
        """최근 체결 내역 조회"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/trades/ticks?market={market}&count={count}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"체결 내역 조회 실패: {str(e)}")

    async def get_ticker(self, markets):
        """현재가 정보 조회"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/ticker?markets={markets}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"현재가 조회 실패: {str(e)}")

    async def get_orderbook(self, markets):
        """호가 정보 조회"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/orderbook?markets={markets}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"호가 조회 실패: {str(e)}")

    def _generate_token(self, query=None):
        """JWT 토큰 생성"""
        payload = {
            'access_key': self.connect_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': round(time.time() * 1000)
        }
        
        if query:
            hash = hashlib.sha512()
            hash.update(query.encode())
            payload['query_hash'] = hash.hexdigest()
            payload['query_hash_alg'] = 'SHA512'
            
        jwt_token = jwt.encode(payload, self.secret_key)
        return f'Bearer {jwt_token}'
    
    async def fetch_market_data(self, symbol='ADA'):  # KRW-ADA -> ADA로 변경
        """현재 시장 데이터 조회"""
        endpoint = f"/public/ticker/{symbol}"  # 심볼만 사용
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == '0000':
                return {
                    'price': float(data['data']['closing_price']),
                    'volume': float(data['data']['units_traded_24H']),
                    'timestamp': datetime.now()
                }
            else:
                raise Exception(f"API 에러: {data['message']}")
                
        except Exception as e:
            raise Exception(f"시장 데이터 조회 실패: {str(e)}")

    async def get_balance(self):
        """계좌 잔고 조회"""
        headers = {'Authorization': self._generate_token()}
        
        try:
            response = requests.get(f"{self.base_url}/v1/accounts", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"잔고 조회 실패: {str(e)}")

    async def place_order(self, symbol, side, volume, price, ord_type='limit'):
        """주문 실행"""
        request_body = {
            'market': f'KRW-{symbol}',
            'side': side,  # bid(매수) / ask(매도)
            'volume': volume,
            'price': price,
            'ord_type': ord_type
        }
        
        query = urlencode(request_body)
        headers = {
            'Authorization': self._generate_token(query),
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/orders",
                data=json.dumps(request_body),
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"주문 실패: {str(e)}")

    async def get_order(self, uuid):
        """주문 조회"""
        params = {'uuid': uuid}
        query = urlencode(params)
        headers = {'Authorization': self._generate_token(query)}
        
        try:
            response = requests.get(
                f"{self.base_url}/v1/order",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"주문 조회 실패: {str(e)}")

    async def cancel_order(self, uuid):
        """주문 취소"""
        params = {'uuid': uuid}
        query = urlencode(params)
        headers = {'Authorization': self._generate_token(query)}
        
        try:
            response = requests.delete(
                f"{self.base_url}/v1/order",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"주문 취소 실패: {str(e)}") 