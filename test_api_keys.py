from bithumb_api import BithumbAPI
import json
import os
from dotenv import load_dotenv

def test_api_keys():
    """
    API 키 리스트 조회 테스트
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
    
    print("\n=== API 키 리스트 조회 테스트 ===")
    print("주의: API Key와 Secret Key가 설정되어 있어야 합니다.")
    
    try:
        result = api.get_api_keys()
        print("\nAPI 키 정보:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # API 키 만료 정보 확인
        if isinstance(result, list):
            for key_info in result:
                print(f"\nAPI Key: {key_info.get('access_key')}")
                print(f"만료일시: {key_info.get('expire_at')}")
        
        return True
    except Exception as e:
        print(f"API 키 리스트 조회 실패: {str(e)}")
        return False

if __name__ == "__main__":
    print("API 키 리스트 조회 테스트 시작...")
    success = test_api_keys()
    
    print("\n=== 테스트 결과 ===")
    print(f"API 키 리스트 조회: {'성공' if success else '실패'}") 