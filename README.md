# 암호화폐 거래 분석 및 테스트 도구

이 프로젝트는 빗썸 거래소의 암호화폐 거래 데이터를 분석하고 실제 거래를 테스트할 수 있는 도구입니다.

## 프로젝트 구조

### 1. 데이터 분석 (notebooks/)
- `1_data_exploration.ipynb`: 초기 데이터 탐색 및 분석
  - RSI와 볼린저 밴드 기반 패턴 분석
  - 거래 성공률 분석
  - 시장 상황별 전략 수립

### 2. API 및 데이터 수집 (src/)
- `bithumb_api.py`: 빗썸 API 연동 클래스
  - 공개 API (시장 데이터 조회)
  - 개인 API (주문, 잔고 조회 등)
- `data_fetcher.py`: 실시간 데이터 수집 도구
  - 호가 데이터 수집
  - 캔들스틱 데이터 수집
  - 거래 기회 분석

### 3. 거래 테스트 도구
- `test_token_order.py`: 단일 토큰 주문 테스트
- `test_bithumb.py`: API 기능 테스트
- `live_backtest.py`: 실시간 거래 전략 테스트
  - RSI 기반 매매 신호
  - 호가 데이터 기반 거래량 결정
  - 리스크 관리

## 설치 및 설정

1. 필요한 패키지 설치:
```bash
pip install requests jwt pandas numpy
```

2. API 키 설정:
- 빗썸 웹사이트에서 API 키 발급
- `bithumb_api.py` 파일에 API 키 입력:
```python
self.connect_key = "YOUR_CONNECT_KEY"
self.secret_key = "YOUR_SECRET_KEY"
```

## 사용 방법

### 1. 데이터 분석
- notebooks 폴더의 Jupyter 노트북 실행
- 기존 거래 패턴 및 성공률 분석
- RSI, 볼린저 밴드 기반 전략 검증

### 2. 실시간 거래 테스트
```bash
python live_backtest.py
```
- 실시간 시장 데이터 모니터링
- RSI 기반 매매 신호 생성
- 거래 기회 포착 및 주문 실행

### 3. 단일 토큰 거래 테스트
```bash
python test_token_order.py
```
- 원하는 토큰 선택
- 수량 설정
- 시장 가격 확인 후 주문

## 주요 기능

### 데이터 수집
- 실시간 호가 데이터
- 분/일/주 단위 캔들스틱
- 체결 내역

### 거래 분석
- RSI 기반 매매 시점 포착
- 호가창 유동성 분석
- 스프레드 및 거래량 분석

### 거래 실행
- 지정가 주문
- 주문 취소
- 잔고 조회

## 주의사항
1. 실제 거래가 이루어지므로 신중하게 사용
2. 테스트는 소액으로 진행 권장
3. API 키는 절대 공개하지 말 것
4. 거래 전략은 충분한 테스트 후 적용

## 에러 해결
- "API 키 오류": API 키 확인
- "잔고 부족": 계좌 잔고 확인
- "가격 범위 오류": 주문 가격 범위 확인