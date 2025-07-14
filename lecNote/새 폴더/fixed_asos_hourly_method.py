def fetch_asos_hourly_data(self):
    """ASOS 시간자료 API 호출 - 수정된 버전"""
    try:
        station = self.seoul_stations['main']
        
        # 🔧 수정: 현재 시간 기준으로 가장 최근 데이터 조회
        now = datetime.now()
        
        # 현재 시간보다 1-2시간 이전 데이터 (API 지연 고려)
        target_time = now - timedelta(hours=2)
        
        # 시간별 데이터이므로 정확한 시간 포맷 사용
        start_dt = target_time.strftime('%Y%m%d')
        start_hour = target_time.strftime('%H')
        
        params = {
            'serviceKey': self.service_key,
            'pageNo': '1',
            'numOfRows': '24',  # 하루치 데이터 조회
            'dataType': 'JSON',
            'dataCd': 'ASOS',
            'dateCd': 'HR',
            'startDt': start_dt,
            'endDt': start_dt,  # 같은 날짜
            'startHh': start_hour,
            'endHh': '23',  # 해당 시간부터 23시까지
            'stnIds': station['stnId']
        }
        
        response = requests.get(self.apis['asos_hourly']['url'], params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'response' in data and data['response']['header']['resultCode'] == '00':
                items = data['response']['body'].get('items', {}).get('item', [])
                
                if items:
                    # 가장 최근 데이터 선택
                    if isinstance(items, list):
                        # 시간순 정렬 후 최신 데이터 선택
                        items.sort(key=lambda x: x.get('tm', ''), reverse=True)
                        item = items[0]
                    else:
                        item = items
                    
                    return {
                        'success': True,
                        'data': {
                            'temperature': float(item.get('ta', 0) or 0),
                            'precipitation': float(item.get('rn', 0) or 0),
                            'humidity': float(item.get('hm', 60) or 60),
                            'wind_speed': float(item.get('ws', 0) or 0),
                            'pressure': float(item.get('ps', 1013) or 1013),
                            'observation_time': item.get('tm', ''),
                            'station_name': station['name'],
                            'data_quality': 'HIGH'
                        }
                    }
                else:
                    return {'success': False, 'error': '해당 시간대 데이터 없음'}
            else:
                # 상세한 오류 정보
                error_code = data.get('response', {}).get('header', {}).get('resultCode', 'UNKNOWN')
                error_msg = data.get('response', {}).get('header', {}).get('resultMsg', '알 수 없는 오류')
                return {'success': False, 'error': f'API 오류 ({error_code}): {error_msg}'}
        else:
            return {'success': False, 'error': f'HTTP {response.status_code}'}
            
    except Exception as e:
        return {'success': False, 'error': f'예외 발생: {str(e)}'}