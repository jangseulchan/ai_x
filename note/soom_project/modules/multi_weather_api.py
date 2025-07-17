# modules/multi_weather_api.py - 4개 기상청 API 통합 시스템
import requests
import urllib.parse
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


class MultiWeatherAPI:
    """4개 기상청 API 통합 관리 클래스"""
    
    def __init__(self, service_key):
        # URL 디코딩 (공공데이터포털 키는 보통 인코딩되어 제공됨)
        self.service_key = urllib.parse.unquote(service_key)
        
        # 4개 주요 API 엔드포인트
        self.apis = {
            'asos_hourly': {
                'url': 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList',
                'name': '지상(ASOS) 시간자료',
                'description': '실시간 관측소 데이터 (정확한 강수량, 온도, 습도)'
            },
            'asos_daily': {
                'url': 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList', 
                'name': '지상(ASOS) 일자료',
                'description': '일별 종합 기상 데이터 (누적 강수량, 최고/최저 온도)'
            },
            'weather_warning': {
                'url': 'http://apis.data.go.kr/1360000/WthrWrnInfoService/getWthrWrnList',
                'name': '기상특보',
                'description': '호우경보, 대설경보 등 기상특보 (침수 위험 직접 지표)'
            },
            'short_forecast': {
                'url': 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst',
                'name': '단기예보(초단기실황)',
                'description': '격자 기반 실시간 데이터'
            }
        }
        
        # 서울 관측소 정보
        self.seoul_stations = {
            'main': {'stnId': '108', 'name': '서울', 'nx': 60, 'ny': 127},
            'gangnam': {'stnId': '401', 'name': '서울강남', 'nx': 61, 'ny': 125},
            'songpa': {'stnId': '402', 'name': '서울송파', 'nx': 62, 'ny': 126}
        }
        
    def get_comprehensive_weather_data(self):
        """4개 API에서 종합 기상 데이터 수집"""
        print("🌤️ 4개 API 종합 데이터 수집 시작...")
        
        results = {
            'timestamp': datetime.now(),
            'data_sources': [],
            'weather_data': {},
            'warnings': [],
            'forecast': {},
            'success': False,
            'errors': []
        }
        
        # 1. ASOS 시간자료 (가장 정확한 실시간 데이터)
        asos_hourly = self.fetch_asos_hourly_data()
        if asos_hourly['success']:
            results['weather_data'].update(asos_hourly['data'])
            results['data_sources'].append('ASOS 시간자료')
            print("✅ ASOS 시간자료 수집 성공")
        else:
            results['errors'].append(f"ASOS 시간자료: {asos_hourly['error']}")
            print(f"❌ ASOS 시간자료 실패: {asos_hourly['error']}")
        
        # 2. ASOS 일자료 (누적/통계 데이터)
        asos_daily = self.fetch_asos_daily_data()
        if asos_daily['success']:
            results['weather_data'].update(asos_daily['data'])
            results['data_sources'].append('ASOS 일자료')
            print("✅ ASOS 일자료 수집 성공")
        else:
            results['errors'].append(f"ASOS 일자료: {asos_daily['error']}")
            print(f"❌ ASOS 일자료 실패: {asos_daily['error']}")
        
        # 3. 기상특보 (침수 직접 경보)
        weather_warning = self.fetch_weather_warnings()
        if weather_warning['success']:
            results['warnings'] = weather_warning['data']
            results['data_sources'].append('기상특보')
            print(f"✅ 기상특보 수집 성공 ({len(weather_warning['data'])}건)")
        else:
            results['errors'].append(f"기상특보: {weather_warning['error']}")
            print(f"❌ 기상특보 실패: {weather_warning['error']}")
        
        # 4. 단기예보 (격자 데이터)
        short_forecast = self.fetch_short_forecast_data()
        if short_forecast['success']:
            results['forecast'] = short_forecast['data']
            results['data_sources'].append('단기예보')
            print("✅ 단기예보 수집 성공")
        else:
            results['errors'].append(f"단기예보: {short_forecast['error']}")
            print(f"❌ 단기예보 실패: {short_forecast['error']}")
        
        # 성공 여부 판단 (최소 1개 API 성공)
        results['success'] = len(results['data_sources']) > 0
        
        if results['success']:
            # 데이터 통합 및 보완
            results['weather_data'] = self.integrate_weather_data(results)
            print(f"🎯 종합 데이터 수집 완료! (성공: {len(results['data_sources'])}/4)")
        else:
            print("❌ 모든 API 호출 실패")
        
        return results
    
    def fetch_asos_hourly_data(self):
        """ASOS 시간자료 API 호출"""
        try:
            station = self.seoul_stations['main']
            yesterday = datetime.now() - timedelta(days=1)
            tm = yesterday.strftime('%Y%m%d23')
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '10',
                'dataType': 'JSON',
                'dataCd': 'ASOS',
                'dateCd': 'HR',
                'startDt': tm,
                'endDt': tm,
                'stnIds': station['stnId']
            }
            
            response = requests.get(self.apis['asos_hourly']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'response' in data and data['response']['header']['resultCode'] == '00':
                    items = data['response']['body'].get('items', {}).get('item', [])
                    
                    if items:
                        item = items[0] if isinstance(items, list) else items
                        
                        return {
                            'success': True,
                            'data': {
                                'temperature': float(item.get('ta', 0) or 0),  # 기온
                                'precipitation': float(item.get('rn', 0) or 0),  # 강수량
                                'humidity': float(item.get('hm', 60) or 60),  # 습도
                                'wind_speed': float(item.get('ws', 0) or 0),  # 풍속
                                'pressure': float(item.get('ps', 1013) or 1013),  # 기압
                                'observation_time': item.get('tm', tm),
                                'station_name': station['name'],
                                'data_quality': 'HIGH'  # ASOS는 고품질 데이터
                            }
                        }
                    else:
                        return {'success': False, 'error': '데이터 없음'}
                else:
                    return {'success': False, 'error': data['response']['header'].get('resultMsg', '응답 오류')}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def fetch_asos_daily_data(self):
        """ASOS 일자료 API 호출"""
        try:
            station = self.seoul_stations['main']            
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '10',
                'dataType': 'JSON',
                'dataCd': 'ASOS',
                'dateCd': 'DAY',
                'startDt': yesterday,
                'endDt': yesterday,
                'stnIds': station['stnId']
            }
            
            response = requests.get(self.apis['asos_daily']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'response' in data and data['response']['header']['resultCode'] == '00':
                    items = data['response']['body'].get('items', {}).get('item', [])
                    
                    if items:
                        # 최신 일자료 선택
                        latest_item = items[-1] if isinstance(items, list) else items
                        
                        return {
                            'success': True,
                            'data': {
                                'daily_precipitation': float(latest_item.get('sumRn', 0) or 0),  # 일강수량
                                'max_temperature': float(latest_item.get('maxTa', 0) or 0),  # 최고기온
                                'min_temperature': float(latest_item.get('minTa', 0) or 0),  # 최저기온
                                'avg_humidity': float(latest_item.get('avgRhm', 60) or 60),  # 평균습도
                                'max_wind_speed': float(latest_item.get('maxWs', 0) or 0),  # 최대풍속
                                'sunshine_duration': float(latest_item.get('sumSsHr', 0) or 0),  # 일조시간
                                'observation_date': latest_item.get('tm', yesterday)
                            }
                        }
                    else:
                        return {'success': False, 'error': '일자료 없음'}
                else:
                    return {'success': False, 'error': data['response']['header'].get('resultMsg', '응답 오류')}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def fetch_weather_warnings(self):
        """기상특보 API 호출 (호우경보 등)"""
        try:
            now = datetime.now()
            from_tm = (now - timedelta(hours=6)).strftime('%Y%m%d%H%M')  # 6시간 전부터
            to_tm = now.strftime('%Y%m%d%H%M')
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '100',
                'dataType': 'JSON',
                'fromTmFc': from_tm,
                'toTmFc': to_tm,
                'stnId': '108'  # 서울
            }
            
            response = requests.get(self.apis['weather_warning']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'response' in data and data['response']['header']['resultCode'] == '00':
                    items = data['response']['body'].get('items', {}).get('item', [])
                    
                    warnings = []
                    if items:
                        item_list = items if isinstance(items, list) else [items]
                        
                        for item in item_list:
                            warning_type = item.get('wrn', '')
                            
                            # 침수 관련 특보만 필터링
                            if any(keyword in warning_type for keyword in ['호우', '태풍', '강풍', '풍랑']):
                                warnings.append({
                                    'type': warning_type,
                                    'level': item.get('lvl', ''),
                                    'issued_time': item.get('tmFc', ''),
                                    'region': item.get('reg', ''),
                                    'content': item.get('cn', ''),
                                    'flood_risk_factor': True  # 침수 위험 요인
                                })
                    
                    return {
                        'success': True,
                        'data': warnings
                    }
                else:
                    return {'success': False, 'error': data['response']['header'].get('resultMsg', '특보 없음')}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def fetch_short_forecast_data(self):
        """단기예보 (격자) API 호출"""
        try:
            station = self.seoul_stations['main']
            now = datetime.now()
            base_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '1000',
                'dataType': 'JSON',
                'base_date': base_time.strftime('%Y%m%d'),
                'base_time': base_time.strftime('%H%M'),
                'nx': str(station['nx']),
                'ny': str(station['ny'])
            }
            
            response = requests.get(self.apis['short_forecast']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'response' in data and data['response']['header']['resultCode'] == '00':
                    items = data['response']['body'].get('items', {}).get('item', [])
                    
                    forecast_data = {}
                    for item in items:
                        category = item.get('category')
                        value = item.get('obsrValue', 0)
                        
                        if category == 'T1H':  # 기온
                            forecast_data['grid_temperature'] = float(value)
                        elif category == 'RN1':  # 1시간 강수량
                            forecast_data['grid_precipitation'] = float(value)
                        elif category == 'REH':  # 습도
                            forecast_data['grid_humidity'] = float(value)
                        elif category == 'WSD':  # 풍속
                            forecast_data['grid_wind_speed'] = float(value)
                    
                    return {
                        'success': True,
                        'data': forecast_data
                    }
                else:
                    return {'success': False, 'error': data['response']['header'].get('resultMsg', '예보 없음')}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def integrate_weather_data(self, results):
        """4개 API 데이터 통합 및 보완"""
        integrated = results['weather_data'].copy()
        
        # 강수량 데이터 우선순위: ASOS 시간자료 > 격자 데이터
        precipitation = (
            integrated.get('precipitation') or 
            integrated.get('grid_precipitation') or 
            integrated.get('daily_precipitation', 0) / 24 or  # 일강수량을 시간당으로 변환
            0
        )
        
        # 온도 데이터 우선순위: ASOS > 격자
        temperature = (
            integrated.get('temperature') or 
            integrated.get('grid_temperature') or 
            (integrated.get('max_temperature', 20) + integrated.get('min_temperature', 10)) / 2 or
            20
        )
        
        # 습도 데이터 우선순위
        humidity = (
            integrated.get('humidity') or 
            integrated.get('grid_humidity') or 
            integrated.get('avg_humidity') or 
            60
        )
        
        # 풍속 데이터
        wind_speed = (
            integrated.get('wind_speed') or 
            integrated.get('grid_wind_speed') or 
            integrated.get('max_wind_speed', 0) / 2 or  # 최대풍속의 절반으로 추정
            0
        )
        
        # 기상특보 기반 위험도 가중치
        warning_risk_factor = 1.0
        if results['warnings']:
            for warning in results['warnings']:
                if '호우' in warning['type']:
                    if '경보' in warning['level']:
                        warning_risk_factor = 2.5  # 호우경보
                    elif '주의보' in warning['level']:
                        warning_risk_factor = 1.8  # 호우주의보
                elif '태풍' in warning['type']:
                    warning_risk_factor = 3.0  # 태풍
        
        # 통합된 최종 데이터
        final_data = {
            'precipitation': precipitation,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': integrated.get('pressure', 1013),
            'warning_risk_factor': warning_risk_factor,
            'data_quality_score': len(results['data_sources']) / 4 * 100,  # 데이터 품질 점수
            'active_warnings': len(results['warnings']),
            'data_sources_used': results['data_sources']
        }
        
        return final_data

def test_multi_weather_api():
    """4개 API 통합 테스트"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    service_key = os.getenv('OPENWEATHER_API_KEY')  # 실제로는 data.go.kr 키
    
    if not service_key:
        print("❌ 서비스 키가 없습니다!")
        return False
    
    api = MultiWeatherAPI(service_key)
    
    print("🇰🇷 4개 기상청 API 통합 테스트")
    print("=" * 50)
    
    # 종합 데이터 수집
    results = api.get_comprehensive_weather_data()
    
    if results['success']:
        data = results['weather_data']
        print(f"\n✅ 종합 데이터 수집 성공!")
        print(f"📊 사용된 API: {', '.join(results['data_sources'])}")
        print(f"📈 데이터 품질: {data['data_quality_score']:.1f}%")
        print(f"\n🌡️ 통합 기상 정보:")
        print(f"   🌧️ 강수량: {data['precipitation']:.1f}mm")
        print(f"   🌡️ 온도: {data['temperature']:.1f}°C")
        print(f"   💧 습도: {data['humidity']:.1f}%")
        print(f"   💨 풍속: {data['wind_speed']:.1f}m/s")
        print(f"   📊 기압: {data['pressure']:.1f}hPa")
        print(f"   ⚠️ 특보 위험계수: {data['warning_risk_factor']:.1f}x")
        print(f"   🚨 활성 특보: {data['active_warnings']}건")
        
        if results['warnings']:
            print(f"\n🚨 현재 기상특보:")
            for warning in results['warnings']:
                print(f"   - {warning['type']} {warning['level']} ({warning['region']})")
        
        return True
    else:
        print(f"\n❌ 데이터 수집 실패:")
        for error in results['errors']:
            print(f"   - {error}")
        return False

if __name__ == "__main__":
    test_multi_weather_api()