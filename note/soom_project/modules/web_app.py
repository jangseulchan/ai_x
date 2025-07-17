# modules/web_app.py
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from datetime import datetime, timedelta
import io
import base64
import time
import threading
import requests
import urllib.parse
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from modules.multi_weather_api import MultiWeatherAPI
import warnings
warnings.filterwarnings('ignore')

class FloodWebApp:
    """4개 기상청 API 통합 침수 예측 시스템 - 3년치 데이터 + 한시간마다 업데이트""" 

    def __init__(self):
        # .env 파일 로드
        load_dotenv()
        self.app = Flask(__name__)
        
        # 추가됨
        # ✅ 세션 유지 위해 필수!
        self.app.secret_key = 'soom'
        
        # Flask 설정: 정적 파일과 템플릿 경로 설정
        self.app.static_folder = os.path.join(os.path.dirname(__file__), 'static')
        self.app.template_folder = os.path.join(os.path.dirname(__file__), 'templates')
        
        self.model = None
        self.feature_names = []
        self.data = None
        self.model_loaded = False
        self.data_last_updated = None
        self.data_start_date = None
        self.data_end_date = None
        self.auto_update_enabled = False
        self.update_interval = 3600  # 1시간 (3600초)
        self.last_check_time = None
        
        # 4개 기상청 API 통합 설정 (공공데이터포털)
        self.service_key = os.getenv('OPENWEATHER_API_KEY')  # 실제로는 data.go.kr 키
        self.city = os.getenv('WEATHER_CITY', 'Seoul')
        self.nx = int(os.getenv('WEATHER_NX', 60))  # 서울 격자 X
        self.ny = int(os.getenv('WEATHER_NY', 127))  # 서울 격자 Y
        
        # 데이터 저장 경로 설정
        self.data_dir = 'data'
        self.processed_dir = 'data/processed'
        self.raw_dir = 'data/raw'
        self.ensure_directories()
        
        # API 키 확인 및 4개 API 통합 객체 생성
        if self.service_key:
            self.service_key = urllib.parse.unquote(self.service_key)
            self.multi_api = MultiWeatherAPI(self.service_key)
            self.api_available = True
            print(f"✅ 4개 기상청 API 키 설정됨 - 위치: {self.city}")
            print("📡 사용 API: ASOS시간자료, ASOS일자료, 기상특보, 단기예보")
        else:
            print("⚠️ .env 파일에 OPENWEATHER_API_KEY를 설정해주세요!")
            print("🔗 https://data.go.kr 에서 기상청 API 4개 서비스 키 발급 가능")
            self.api_available = False
            self.multi_api = None
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        self.setup_routes()
        self.check_initial_data()
        self.start_auto_update_service()
    
    # 추가됨
    def setup_routes(self):
        """모든 라우트 설정"""
        # 추가됨
        @self.app.route('/api/flood_map')
        def flood_map():
            if self.data is None or len(self.data) == 0:
                return jsonify({})
            
            # 최근 1시간 데이터 필터링
            now = datetime.now()
            recent_time = now - timedelta(hours=1)
            df_recent = self.data[self.data['obs_date'] >= recent_time]

            if df_recent.empty:
                df_recent = self.data.tail(1)  # fallback

            avg_precip = df_recent['precipitation'].mean()
            avg_humidity = df_recent['humidity'].mean()
            month = now.month
            season_score = 1.0 if month in [6,7,8,9] else 0.3

            warning_factor = df_recent['warning_risk_factor'].mean() if 'warning_risk_factor' in df_recent else 1.0

            # 위험도 계산
            score = (avg_precip * 0.4 + avg_humidity * 0.3 + season_score * 100 * 0.3) * warning_factor
            score = min(100, round(score, 1))

            # 서울 25개구 이름 목록
            seoul_gu = [
                '강남구','강동구','강북구','강서구','관악구','광진구','구로구','금천구','노원구','도봉구',
                '동대문구','동작구','마포구','서대문구','서초구','성동구','성북구','송파구','양천구','영등포구',
                '용산구','은평구','종로구','중구','중랑구'
            ]

            return jsonify({gu: score for gu in seoul_gu})

    def ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [self.data_dir, self.processed_dir, self.raw_dir, 'models', 'outputs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def check_initial_data(self):
        """초기 데이터 및 모델 확인"""
        # 데이터 확인
        data_paths = [
            f'{self.processed_dir}/ML_COMPLETE_DATASET.csv',
            'STRATEGIC_FLOOD_DATA/4_ML_READY/ML_COMPLETE_DATASET.csv',
            'ML_COMPLETE_DATASET.csv'
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                try:
                    self.data = pd.read_csv(path)
                    if 'obs_date' in self.data.columns:
                        self.data['obs_date'] = pd.to_datetime(self.data['obs_date'])
                        self.data_start_date = self.data['obs_date'].min()
                        self.data_end_date = self.data['obs_date'].max()
                    
                    # 오늘까지 자동 채우기
                    self.fill_to_today()
                    
                    self.data_last_updated = datetime.now()
                    print(f"✅ 데이터 발견: {path}")
                    print(f"📅 데이터 기간: {self.data_start_date} ~ {self.data_end_date}")
                    print(f"📊 총 데이터: {len(self.data)}행")
                    break
                except:
                    continue
        
        # 데이터가 없으면 정확히 3년치 생성
        if self.data is None:
            print("📊 기존 데이터가 없습니다. 정확히 3년치 샘플 데이터를 생성합니다...")
            self.generate_3year_sample_data()
        
        # 모델 확인
        if os.path.exists('models/randomforest_model.pkl'):
            try:
                self.model = joblib.load('models/randomforest_model.pkl')
                self.feature_names = joblib.load('models/feature_names.pkl')
                self.model_loaded = True
                print("✅ 모델 로드 성공")
            except:
                print("❌ 모델 로드 실패")
    
    def generate_3year_sample_data(self):
        """정확히 3년치 샘플 데이터 생성 (1,095일)"""
        print("🏗️ 정확히 3년치 기상 데이터 생성 중...")
        
        # 정확히 3년 = 1095일 (365*3) - 윤년 고려하면 1096일
        target_days = 1096  # 윤년 포함
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=target_days-1)  # -1은 오늘 포함
        
        print(f"📅 데이터 범위: {start_date.date()} ~ {end_date.date()}")
        print(f"🎯 목표 일수: {target_days}일")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D', inclusive='both')
        
        data_list = []
        for date in date_range:
            # 계절별 현실적인 데이터 생성
            month = date.month
            
            if month in [6, 7, 8, 9]:  # 장마철/여름
                precipitation = np.random.exponential(8) * np.random.choice([0, 1, 1, 1, 2, 3, 5], p=[0.6, 0.15, 0.1, 0.08, 0.04, 0.02, 0.01])
                humidity = np.clip(np.random.normal(75, 12), 50, 95)
                avg_temp = np.clip(np.random.normal(26, 4), 20, 35)
            elif month in [12, 1, 2]:  # 겨울
                precipitation = np.random.exponential(2) * np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
                humidity = np.clip(np.random.normal(55, 15), 30, 80)
                avg_temp = np.clip(np.random.normal(2, 6), -15, 15)
            else:  # 봄/가을
                precipitation = np.random.exponential(4) * np.random.choice([0, 1, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.04, 0.01])
                humidity = np.clip(np.random.normal(65, 15), 40, 85)
                avg_temp = np.clip(np.random.normal(15, 8), 5, 25)
            
            # 극한 기상 현상 추가 (드물게)
            if np.random.random() < 0.01:  # 1% 확률로 극한 강수
                precipitation = np.random.uniform(80, 200)
            
            row = {
                'obs_date': date,
                'precipitation': max(0, precipitation),
                'humidity': humidity,
                'avg_temp': avg_temp,
                'wind_speed': max(0, np.random.normal(3, 2)),
                'month': month,
                'precip_ma3': 0,  # 나중에 계산
                'precip_ma7': 0,  # 나중에 계산
                'is_peak_rainy': 1 if month in [6, 7, 8, 9] else 0,
                'precip_risk_level': self.get_precip_level(precipitation),
                'is_flood_risk': 1 if precipitation >= 50 else 0,
                
                # 4개 API 통합 관련 필드 추가
                'data_quality_score': 100,  # 샘플 데이터는 100점
                'warning_risk_factor': 1.0,  # 기본 위험 계수
                'active_warnings': 0,  # 특보 없음
                'pressure': np.clip(np.random.normal(1013, 10), 980, 1040),
                'api_sources': 'generated'
            }
            data_list.append(row)
        
        self.data = pd.DataFrame(data_list)
        
        # 이동평균 계산
        self.data['precip_ma3'] = self.data['precipitation'].rolling(window=3, min_periods=1).mean()
        self.data['precip_ma7'] = self.data['precipitation'].rolling(window=7, min_periods=1).mean()
        
        # 누적 강우일수 계산
        self.data['rain_days_cumsum'] = (self.data['precipitation'] > 0).cumsum()
        
        self.data_start_date = self.data['obs_date'].min()
        self.data_end_date = self.data['obs_date'].max()
        self.data_last_updated = datetime.now()
        
        # 데이터 저장
        output_path = f'{self.processed_dir}/ML_COMPLETE_DATASET.csv'
        self.data.to_csv(output_path, index=False)
        
        print(f"✅ 정확히 3년치 데이터 생성 완료!")
        print(f"📊 총 {len(self.data)}행 생성 (목표: {target_days}일)")
        print(f"📅 기간: {self.data_start_date.date()} ~ {self.data_end_date.date()}")
        print(f"💾 저장: {output_path}")
        print(f"🌧️ 침수 위험일: {self.data['is_flood_risk'].sum()}일 ({self.data['is_flood_risk'].mean()*100:.1f}%)")
    
    def fill_to_today(self):
        """마지막 데이터부터 오늘까지 자동 채우기"""
        if self.data is None or len(self.data) == 0:
            return
        
        last_date = self.data_end_date.date() if self.data_end_date else datetime.now().date() - timedelta(days=10)
        today = datetime.now().date()
        
        current_date = last_date + timedelta(days=1)
        added_count = 0
        
        while current_date <= today:
            # 계절별 현실적인 데이터 생성
            if current_date.month in [6, 7, 8, 9]:
                precipitation = max(0, np.random.exponential(8))  # 장마철
                humidity = np.clip(np.random.normal(75, 12), 50, 95)
                avg_temp = np.clip(np.random.normal(26, 4), 20, 32)
            else:
                precipitation = max(0, np.random.exponential(3))  # 평상시
                humidity = np.clip(np.random.normal(65, 15), 30, 90)
                avg_temp = np.clip(np.random.normal(24, 6), 15, 35)
            
            new_row = {
                'obs_date': pd.Timestamp(current_date),
                'precipitation': precipitation,
                'humidity': humidity,
                'avg_temp': avg_temp,
                'wind_speed': max(0, np.random.normal(3, 2)),
                'month': current_date.month,
                'precip_ma3': precipitation,
                'precip_ma7': precipitation,
                'is_peak_rainy': 1 if current_date.month in [6, 7, 8, 9] else 0,
                'precip_risk_level': self.get_precip_level(precipitation),
                'is_flood_risk': 1 if precipitation >= 50 else 0,
                
                # 4개 API 관련 필드
                'data_quality_score': 50,  # 추정 데이터
                'warning_risk_factor': 1.0,
                'active_warnings': 0,
                'pressure': 1013,
                'api_sources': 'filled'
            }
            
            new_df = pd.DataFrame([new_row])
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            current_date += timedelta(days=1)
            added_count += 1
        
        if added_count > 0:
            self.data_end_date = pd.Timestamp(today)
            # 데이터 저장
            self.save_data()
            print(f"📅 오늘까지 데이터 채움: +{added_count}일 (총 {len(self.data)}행)")
    
    def save_data(self):
        """데이터 저장"""
        output_path = f'{self.processed_dir}/ML_COMPLETE_DATASET.csv'
        self.data.to_csv(output_path, index=False)
        # print(f"💾 데이터 저장: {output_path}")  # 너무 많은 로그 방지
    
    def start_auto_update_service(self):
        """자동 업데이트 서비스 시작 (1시간마다)"""
        def auto_update_worker():
            while True:
                if self.auto_update_enabled:
                    self.last_check_time = datetime.now()
                    try:
                        print(f"🕐 {self.last_check_time.strftime('%H:%M:%S')} - 4개 API 자동 업데이트 실행")
                        
                        # 4개 기상청 API 통합 호출
                        if self.api_available and self.multi_api:
                            self.multi_weather_update()
                        else:
                            self.simulate_data_update()
                        
                        # 데이터 저장
                        self.save_data()
                        
                    except Exception as e:
                        print(f"자동 업데이트 오류: {e}")
                
                time.sleep(self.update_interval)  # 1시간 대기
        
        update_thread = threading.Thread(target=auto_update_worker, daemon=True)
        update_thread.start()
        print(f"🔄 자동 업데이트 서비스 시작 (간격: {self.update_interval//60}분)")
    
    def simulate_data_update(self):
        """데이터 업데이트 시뮬레이션"""
        if self.data is not None and len(self.data) > 0:
            # 최신 날짜 이후의 가상 데이터 추가
            last_date = self.data_end_date if self.data_end_date else datetime.now() - timedelta(days=1)
            new_date = last_date + timedelta(hours=1)
            
            # 새로운 데이터 행 생성 (현실적인 기상 데이터)
            precipitation = max(0, np.random.exponential(scale=5))
            
            new_row = {
                'obs_date': new_date,
                'precipitation': precipitation,
                'humidity': np.clip(np.random.normal(70, 15), 20, 100),  # 습도
                'avg_temp': np.clip(np.random.normal(22, 8), -10, 40),   # 온도
                'wind_speed': np.clip(np.random.exponential(scale=3), 0, 20),
                'month': new_date.month,
                'precip_ma3': precipitation,
                'precip_ma7': precipitation,
                'is_peak_rainy': 1 if new_date.month in [6, 7, 8, 9] else 0,
                'precip_risk_level': self.get_precip_level(precipitation),
                'is_flood_risk': 1 if precipitation >= 50 else 0,
                
                # 시뮬레이션 데이터 품질
                'data_quality_score': 25,  # 시뮬레이션은 낮은 품질
                'warning_risk_factor': 1.0,
                'active_warnings': 0,
                'pressure': 1013,
                'api_sources': 'simulation'
            }
            
            # 데이터프레임에 추가
            new_df = pd.DataFrame([new_row])
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            
            # 날짜 정보 업데이트
            self.data_end_date = new_date
            self.data_last_updated = datetime.now()
            
            print(f"🔄 시뮬레이션 업데이트: {new_date} (총 {len(self.data)}행)")
    
    def multi_weather_update(self):
        """4개 기상청 API 통합 실시간 데이터 업데이트"""
        try:
            if not self.multi_api:
                print("❌ 멀티 API 객체가 없습니다")
                self.simulate_data_update()
                return
                
            # 4개 API 종합 데이터 수집
            weather_results = self.multi_api.get_comprehensive_weather_data()
            
            if weather_results['success']:
                new_row = self.process_multi_weather_data(weather_results)
                
                # 데이터프레임에 추가
                new_df = pd.DataFrame([new_row])
                self.data = pd.concat([self.data, new_df], ignore_index=True)
                
                # 날짜 정보 업데이트
                self.data_end_date = new_row['obs_date']
                self.data_last_updated = datetime.now()
                
                api_info = f"({len(weather_results['data_sources'])}/4 성공)"
                print(f"🇰🇷 4개 API 통합 업데이트: {new_row['obs_date']} {api_info}")
                print(f"   📊 강수량: {new_row['precipitation']:.1f}mm, 온도: {new_row['avg_temp']:.1f}°C")
                print(f"   📡 사용 API: {', '.join(weather_results['data_sources'])}")
                print(f"   📈 품질: {new_row['data_quality_score']:.0f}%")
                
                if weather_results['warnings']:
                    print(f"   🚨 기상특보: {len(weather_results['warnings'])}건 활성")
                    print(f"   ⚠️ 위험계수: {new_row['warning_risk_factor']:.1f}x")
                    
            else:
                print("❌ 4개 API 모두 실패 - 시뮬레이션으로 대체")
                self.simulate_data_update()
                
        except Exception as e:
            print(f"4개 API 통합 업데이트 실패: {e}")
            self.simulate_data_update()
    
    def process_multi_weather_data(self, weather_results):
        """4개 API 통합 데이터를 데이터프레임 형식으로 변환"""
        try:
            now = datetime.now()
            data = weather_results['weather_data']
            
            # 기본 기상 데이터
            precipitation = data.get('precipitation', 0)
            temperature = data.get('temperature', 20)
            humidity = data.get('humidity', 60)
            wind_speed = data.get('wind_speed', 0)
            
            # 기상특보 기반 위험도 조정
            warning_factor = data.get('warning_risk_factor', 1.0)
            adjusted_precipitation = precipitation * warning_factor  # 특보시 위험도 증가
            
            return {
                'obs_date': now,
                'precipitation': precipitation,
                'humidity': humidity,
                'avg_temp': temperature,
                'wind_speed': wind_speed,
                'month': now.month,
                'precip_ma3': precipitation,
                'precip_ma7': precipitation,
                'is_peak_rainy': 1 if now.month in [6, 7, 8, 9] else 0,
                'precip_risk_level': self.get_precip_level(adjusted_precipitation),  # 특보 반영
                'is_flood_risk': 1 if adjusted_precipitation >= 50 else 0,
                
                # 4개 API 통합 추가 정보
                'data_quality_score': data.get('data_quality_score', 50),
                'warning_risk_factor': warning_factor,
                'active_warnings': data.get('active_warnings', 0),
                'pressure': data.get('pressure', 1013),
                'api_sources': ','.join(data.get('data_sources_used', ['simulation']))
            }
        except Exception as e:
            print(f"4개 API 데이터 변환 오류: {e}")
            return {
                'obs_date': datetime.now(),
                'precipitation': 0, 'humidity': 60, 'avg_temp': 20, 'wind_speed': 0,
                'month': datetime.now().month, 'precip_ma3': 0, 'precip_ma7': 0,
                'is_peak_rainy': 0, 'precip_risk_level': 0, 'is_flood_risk': 0,
                'data_quality_score': 0, 'warning_risk_factor': 1.0,
                'active_warnings': 0, 'pressure': 1013, 'api_sources': 'error'
            }
    
    # 추가됨
    def log_event(event_type, message):
        os.makedirs('logs', exist_ok=True)
        log_path = 'logs/log_events.json'

        event = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': event_type,
            'message': message
        }
        # 추가됨
        # 기존 로그 불러오기
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.insert(0, event)  # 최근 로그가 위로
        
        # 최대 100개만 저장
        logs = logs[:100]
        # 추가됨
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        # 추가됨
        @self.app.route('/api/log_event', methods=['POST'])
        def log_event_api():
            data = request.get_json()
            log_event(data.get('type', '기타'), data.get('message', ''))
            return jsonify({'success': True})
        # 추가됨
        @self.app.route('/api/get_logs')
        def get_logs():
            log_path = 'logs/log_events.json'
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                return jsonify(logs)
            return jsonify([])
        # 추가됨
        @self.app.route('/register')
        def register_page():
            return render_template('register.html')
        # 추가됨
        @self.app.route('/api/register', methods=['POST'])
        def register_api():
            data = request.get_json()
            username = data['username']
            password = data['password']

            os.makedirs('users', exist_ok=True)
            user_path = 'users/users.json'

            if os.path.exists(user_path):
                with open(user_path, 'r', encoding='utf-8') as f:
                    users = json.load(f)
            else:
                users = {}

            if username in users:
                return jsonify({'success': False, 'message': '이미 존재하는 아이디입니다.'})

            users[username] = password
            with open(user_path, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)

            log_event("회원가입", f"새 사용자 가입: {username}")
            return jsonify({'success': True})

        # 추가됨
        @self.app.route('/models')
        def models_page():
            return render_template('models.html')
        # 추가됨
        @self.app.route('/login')
        def login_page():
            return render_template('login.html')
        # 추가됨
        @self.app.route('/api/login', methods=['POST'])
        def login_api():
            data = request.get_json()
            if data['username'] == 'admin' and data['password'] == '1234':
                session['user'] = 'admin'
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'message': 'ID 또는 비밀번호가 틀립니다.'})
        # 추가됨
        @self.app.route('/api/logout')
        def logout():
            session.pop('user', None)
            return jsonify({'success': True})
        # 추가됨
        @self.app.route('/api/session')
        def session_check():
            return jsonify({'logged_in': 'user' in session})
        # 추가됨
        @self.app.route('/api/select_model', methods=['POST'])
        def select_model():
            data = request.get_json()
            name = data.get('name', '').lower()

            model_path = f'models/{name}_model.pkl'
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.feature_names = joblib.load('models/feature_names.pkl')
                self.model_loaded = True
                self.selected_model_name = name  # 추가
                return jsonify({'success': True, 'message': f'{name} 모델이 설정되었습니다.'})
            else:
                return jsonify({'success': False, 'message': f'{name} 모델 파일이 없습니다.'})

            # 추가됨
        @self.app.route('/api/model_compare')
        def compare_models():
            from modules.data_loader import DataLoader
            from modules.preprocessor import DataPreprocessor
            from modules.trainer import ModelTrainer
            from modules.evaluator import ModelEvaluator

            # 데이터 로드
            loader = DataLoader()
            df = loader.load_ml_ready_data()
            if df is None:
                return jsonify({'success': False, 'message': '데이터를 불러올 수 없습니다.'})

            # 전처리
            pre = DataPreprocessor()
            X, y, features = pre.prepare_features(df)
            if X is None:
                return jsonify({'success': False, 'message': '전처리 실패'})

            # 모델 훈련
            trainer = ModelTrainer()
            models = trainer.train_models(X, y, features)

            # 모델 평가
            evaluator = ModelEvaluator()
            results = evaluator.evaluate_all_models(models, X, y)

            return jsonify({'success': True, 'results': results})

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'data_loaded': self.data is not None,
                'data_rows': len(self.data) if self.data is not None else 0,
                'model_loaded': self.model_loaded,
                'features': len(self.feature_names) if self.feature_names else 0,
                'data_start_date': self.data_start_date.isoformat() if self.data_start_date else None,
                'data_end_date': self.data_end_date.isoformat() if self.data_end_date else None,
                'data_last_updated': self.data_last_updated.isoformat() if self.data_last_updated else None,
                'auto_update_enabled': self.auto_update_enabled,
                'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
                'update_interval': self.update_interval,
                'update_interval_minutes': self.update_interval // 60,
                'api_available': self.api_available,
                'api_location': f"{self.city} (4개 기상청 API 통합)" if self.api_available else None,
                'today': datetime.now().strftime('%Y-%m-%d'),
                'data_file_path': f'{self.processed_dir}/ML_COMPLETE_DATASET.csv',
                'multi_api_enabled': self.multi_api is not None,
                # 추가됨
                'current_model_name': getattr(self, 'selected_model_name', 'randomforest')
            })
        
        @self.app.route('/api/load_data', methods=['POST'])
        def load_data():
            try:
                # 기존 데이터가 있으면 로드, 없으면 3년치 생성
                if self.data is None:
                    self.generate_3year_sample_data()
                
                return jsonify({
                    'success': True,
                    'message': f'데이터 로드 성공: {len(self.data)}행',
                    'rows': len(self.data),
                    'columns': len(self.data.columns),
                    'start_date': self.data_start_date.isoformat() if self.data_start_date else None,
                    'end_date': self.data_end_date.isoformat() if self.data_end_date else None,
                    'file_path': f'{self.processed_dir}/ML_COMPLETE_DATASET.csv',
                    'flood_risk_days': int(self.data['is_flood_risk'].sum()),
                    'flood_risk_percentage': f"{self.data['is_flood_risk'].mean()*100:.1f}%"
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/update_data', methods=['POST'])
        def update_data():
            """수동 데이터 업데이트 (4개 API 통합)"""
            try:
                if self.data is None:
                    return jsonify({'success': False, 'message': '먼저 데이터를 로드하세요.'})
                
                old_count = len(self.data)
                
                # 4개 기상청 API 통합 사용
                if self.api_available and self.multi_api:
                    try:
                        weather_results = self.multi_api.get_comprehensive_weather_data()
                        if weather_results['success']:
                            new_row = self.process_multi_weather_data(weather_results)
                            new_df = pd.DataFrame([new_row])
                            self.data = pd.concat([self.data, new_df], ignore_index=True)
                            
                            # 데이터 저장
                            self.save_data()
                            
                            return jsonify({
                                'success': True,
                                'message': f'4개 API 통합 실제 데이터가 추가되었습니다.',
                                'old_count': old_count,
                                'new_count': len(self.data),
                                'added_count': 1,
                                'latest_date': self.data_end_date.isoformat() if self.data_end_date else None,
                                'data_source': '4개 기상청 API 통합',
                                'api_sources': ', '.join(weather_results['data_sources']),
                                'data_quality': f"{weather_results['weather_data']['data_quality_score']:.1f}%",
                                'precipitation': new_row['precipitation'],
                                'temperature': new_row['avg_temp'],
                                'humidity': new_row['humidity'],
                                'warning_factor': new_row['warning_risk_factor'],
                                'active_warnings': new_row['active_warnings'],
                                'pressure': new_row['pressure'],
                                'saved_to': f'{self.processed_dir}/ML_COMPLETE_DATASET.csv'
                            })
                        else:
                            raise Exception("4개 API 모두 응답 실패")
                    except Exception as e:
                        print(f"4개 API 통합 실패: {e}, 시뮬레이션으로 대체")
                        
                # API 실패 시 시뮬레이션
                for _ in range(np.random.randint(1, 3)):
                    self.simulate_data_update()
                    time.sleep(0.1)
                
                new_count = len(self.data)
                added_count = new_count - old_count
                
                # 데이터 저장
                self.save_data()
                
                return jsonify({
                    'success': True,
                    'message': f'시뮬레이션 데이터 {added_count}개가 추가되었습니다.',
                    'old_count': old_count,
                    'new_count': new_count,
                    'added_count': added_count,
                    'latest_date': self.data_end_date.isoformat() if self.data_end_date else None,
                    'data_source': 'Simulation',
                    'data_quality': '25%',
                    'saved_to': f'{self.processed_dir}/ML_COMPLETE_DATASET.csv'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': f'업데이트 실패: {str(e)}'})
        
        @self.app.route('/api/toggle_auto_update', methods=['POST'])
        def toggle_auto_update():
            """자동 업데이트 토글 (1시간마다)"""
            try:
                self.auto_update_enabled = not self.auto_update_enabled
                
                return jsonify({
                    'success': True,
                    'auto_update_enabled': self.auto_update_enabled,
                    'message': f'4개 API 자동 업데이트가 {"활성화" if self.auto_update_enabled else "비활성화"}되었습니다.',
                    'update_interval_minutes': self.update_interval // 60
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/train_model', methods=['POST'])
        def train_model():
            try:
                if self.data is None:
                    return jsonify({'success': False, 'message': '먼저 데이터를 로드하세요.'})
                
                # 특성 준비 (4개 API 통합 관련 필드 포함)
                basic_features = ['precipitation', 'humidity', 'avg_temp']
                available_features = [col for col in basic_features if col in self.data.columns]
                
                # 추가 특성 (4개 API 통합 특성 포함)
                extra_features = ['wind_speed', 'month', 'precip_ma3', 'precip_ma7', 
                                'is_peak_rainy', 'precip_risk_level', 'pressure',
                                'data_quality_score', 'warning_risk_factor', 'active_warnings']
                for feat in extra_features:
                    if feat in self.data.columns:
                        available_features.append(feat)
                
                # 타겟 변수
                if 'is_flood_risk' not in self.data.columns:
                    self.data['is_flood_risk'] = (self.data['precipitation'] >= 50).astype(int)
                
                X = self.data[available_features]
                y = self.data['is_flood_risk']
                
                # 결측값 처리
                X = X.fillna(X.median())
                
                # 데이터 분할
                split_idx = int(len(X) * 0.8)
                X_train = X.iloc[:split_idx]
                X_test = X.iloc[split_idx:]
                y_train = y.iloc[:split_idx]
                y_test = y.iloc[split_idx:]
                
                # 모델 훈련 (4개 API 통합 특성 반영)
                self.model = RandomForestClassifier(
                    n_estimators=150,  # 트리 수 증가
                    max_depth=20,      # 깊이 증가 (더 많은 특성)
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                
                self.model.fit(X_train, y_train)
                self.feature_names = available_features
                
                # 성능 평가
                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test)[:, 1]
                
                try:
                    auc_score = roc_auc_score(y_test, y_proba)
                except:
                    auc_score = 0.5
                
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # 특성 중요도
                feature_importance = dict(zip(available_features, self.model.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # 모델 저장
                os.makedirs('models', exist_ok=True)
                joblib.dump(self.model, 'models/randomforest_model.pkl')
                joblib.dump(self.feature_names, 'models/feature_names.pkl')
                
                self.model_loaded = True
                
                return jsonify({
                    'success': True,
                    'message': '4개 API 통합 모델 훈련 완료!',
                    'auc': round(auc_score, 3),
                    'precision': round(report['1']['precision'], 3),
                    'recall': round(report['1']['recall'], 3),
                    'f1_score': round(report['1']['f1-score'], 3),
                    'features': len(available_features),
                    'training_data_size': len(X_train),
                    'data_period': f"{self.data_start_date.date()} ~ {self.data_end_date.date()}",
                    'top_features': [f"{feat}: {imp:.3f}" for feat, imp in top_features],
                    'has_warning_features': 'warning_risk_factor' in available_features
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': f'훈련 실패: {str(e)}'})
        
        @self.app.route('/api/create_visualization', methods=['POST'])
        def create_visualization():
            try:
                if self.data is None:
                    return jsonify({'success': False, 'message': '먼저 데이터를 로드하세요.'})
                
                viz_type = request.json.get('type', 'precipitation')
                
                plt.figure(figsize=(14, 10))
                
                if viz_type == 'precipitation':
                    # 강수량 시계열
                    plt.subplot(2, 1, 1)
                    plt.plot(self.data['obs_date'], self.data['precipitation'], alpha=0.7, color='blue')
                    
                    # 4개 API 품질 데이터 강조
                    if 'data_quality_score' in self.data.columns:
                        high_quality = self.data[self.data['data_quality_score'] >= 75]
                        if len(high_quality) > 0:
                            plt.scatter(high_quality['obs_date'], high_quality['precipitation'], 
                                    color='green', s=20, alpha=0.6, label='고품질 데이터 (4개 API)')
                    
                    plt.title('📈 강수량 시계열 분석 (3년) - 4개 API 통합')
                    plt.ylabel('강수량 (mm)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 최근 데이터 강조
                    recent_data = self.data.tail(365)  # 최근 1년
                    plt.subplot(2, 1, 2)
                    plt.plot(recent_data['obs_date'], recent_data['precipitation'], 
                            color='red', linewidth=2, alpha=0.8)
                    
                    # 기상특보 반영 데이터 표시
                    if 'warning_risk_factor' in recent_data.columns:
                        warning_data = recent_data[recent_data['warning_risk_factor'] > 1.0]
                        if len(warning_data) > 0:
                            plt.scatter(warning_data['obs_date'], warning_data['precipitation'], 
                                    color='red', s=50, alpha=0.8, label='기상특보 반영')
                    
                    plt.title('🔍 최근 1년 데이터 (상세)')
                    plt.ylabel('강수량 (mm)')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                
                elif viz_type == 'monthly':
                    # 월별 평균 강수량
                    if 'month' in self.data.columns:
                        monthly_precip = self.data.groupby('month')['precipitation'].agg(['mean', 'std', 'count'])
                        plt.bar(monthly_precip.index, monthly_precip['mean'], 
                                yerr=monthly_precip['std'], alpha=0.8, capsize=5)
                        plt.title('📊 월별 평균 강수량 (3년 평균) - 4개 API 통합')
                        plt.xlabel('월')
                        plt.ylabel('평균 강수량 (mm)')
                        
                        # 데이터 개수 표시
                        for i, count in enumerate(monthly_precip['count']):
                            plt.text(i+1, monthly_precip['mean'].iloc[i] + monthly_precip['std'].iloc[i] + 2, 
                                    f'n={count}', ha='center', fontsize=8)
                
                elif viz_type == 'distribution':
                    # 강수량 분포
                    plt.subplot(2, 2, 1)
                    plt.hist(self.data['precipitation'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50mm 위험선')
                    plt.axvline(x=self.data['precipitation'].mean(), color='green', 
                                linestyle='-', linewidth=2, label=f'평균: {self.data["precipitation"].mean():.1f}mm')
                    plt.title('📊 강수량 분포 (3년)')
                    plt.xlabel('강수량 (mm)')
                    plt.ylabel('빈도')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 데이터 품질 분포
                    if 'data_quality_score' in self.data.columns:
                        plt.subplot(2, 2, 2)
                        plt.hist(self.data['data_quality_score'], bins=20, alpha=0.7, color='orange')
                        plt.title('📈 데이터 품질 분포')
                        plt.xlabel('품질 점수 (%)')
                        plt.ylabel('빈도')
                        plt.grid(True, alpha=0.3)
                    
                    # 기상특보 위험계수 분포
                    if 'warning_risk_factor' in self.data.columns:
                        plt.subplot(2, 2, 3)
                        plt.hist(self.data['warning_risk_factor'], bins=20, alpha=0.7, color='red')
                        plt.title('🚨 기상특보 위험계수 분포')
                        plt.xlabel('위험계수')
                        plt.ylabel('빈도')
                        plt.grid(True, alpha=0.3)
                    
                    # 박스플롯
                    plt.subplot(2, 2, 4)
                    plt.boxplot(self.data['precipitation'], vert=False, patch_artist=True)
                    plt.xlabel('강수량 (mm)')
                    plt.title('📦 강수량 박스플롯')
                    plt.grid(True, alpha=0.3)
                
                elif viz_type == 'correlation':
                    # 상관관계 (4개 API 통합 특성 포함)
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                    # 주요 특성만 선택
                    important_cols = ['precipitation', 'humidity', 'avg_temp', 'wind_speed', 
                                    'pressure', 'data_quality_score', 'warning_risk_factor', 
                                    'active_warnings', 'is_flood_risk']
                    available_cols = [col for col in important_cols if col in numeric_cols][:10]
                    
                    if len(available_cols) > 1:
                        corr_matrix = self.data[available_cols].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                                    square=True, linewidths=0.5, fmt='.2f')
                        plt.title('🔍 변수간 상관관계 (4개 API 통합)')
                
                elif viz_type == 'recent_trend':
                    # 최근 30일 트렌드 (4개 API 통합 정보 포함)
                    if 'obs_date' in self.data.columns and len(self.data) > 0:
                        latest_date = self.data['obs_date'].max()
                        start_date = latest_date - timedelta(days=30)
                        recent_data = self.data[self.data['obs_date'] >= start_date]
                        
                        if len(recent_data) > 0:
                            plt.subplot(4, 1, 1)
                            plt.plot(recent_data['obs_date'], recent_data['precipitation'], 'b-', linewidth=2)
                            # 기상특보 데이터 강조
                            if 'warning_risk_factor' in recent_data.columns:
                                warning_data = recent_data[recent_data['warning_risk_factor'] > 1.0]
                                if len(warning_data) > 0:
                                    plt.scatter(warning_data['obs_date'], warning_data['precipitation'], 
                                                color='red', s=50, alpha=0.8, label='기상특보')
                                    plt.legend()
                            plt.title(f'🕐 최근 30일 강수량 ({len(recent_data)}개 데이터)')
                            plt.ylabel('강수량 (mm)')
                            plt.grid(True, alpha=0.3)
                            
                            plt.subplot(4, 1, 2)
                            plt.plot(recent_data['obs_date'], recent_data['humidity'], 'g-', linewidth=2)
                            plt.title('💧 최근 30일 습도')
                            plt.ylabel('습도 (%)')
                            plt.grid(True, alpha=0.3)
                            
                            plt.subplot(4, 1, 3)
                            plt.plot(recent_data['obs_date'], recent_data['avg_temp'], 'r-', linewidth=2)
                            plt.title('🌡️ 최근 30일 온도')
                            plt.ylabel('온도 (°C)')
                            plt.grid(True, alpha=0.3)
                            
                            # 데이터 품질 추이
                            if 'data_quality_score' in recent_data.columns:
                                plt.subplot(4, 1, 4)
                                plt.plot(recent_data['obs_date'], recent_data['data_quality_score'], 'purple', linewidth=2)
                                plt.fill_between(recent_data['obs_date'], recent_data['data_quality_score'], 
                                                alpha=0.3, color='purple')
                                plt.title('📊 최근 30일 데이터 품질')
                                plt.ylabel('품질 점수 (%)')
                                plt.xlabel('날짜')
                                plt.xticks(rotation=45)
                                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # 이미지를 base64로 변환
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.read()).decode()
                plt.close()
                
                # 추가 통계 정보
                stats_info = {}
                if 'data_quality_score' in self.data.columns:
                    stats_info['avg_quality'] = f"{self.data['data_quality_score'].mean():.1f}%"
                    stats_info['high_quality_ratio'] = f"{(self.data['data_quality_score'] >= 75).mean()*100:.1f}%"
                
                if 'warning_risk_factor' in self.data.columns:
                    stats_info['warning_events'] = int((self.data['warning_risk_factor'] > 1.0).sum())
                
                return jsonify({
                    'success': True,
                    'image': f'data:image/png;base64,{img_base64}',
                    'message': f'{viz_type} 차트 생성 완료 (4개 API 통합)',
                    'data_count': len(self.data),
                    'data_period': f"{self.data_start_date.date()} ~ {self.data_end_date.date()}",
                    'stats': stats_info
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': f'시각화 실패: {str(e)}'})
        
        @self.app.route('/api/predict', methods=['POST'])
        def predict():
            try:
                data = request.get_json()
                
                # 특정 날짜 예측 지원
                target_date = data.get('target_date')
                if target_date:
                    target_date = pd.to_datetime(target_date).date()
                    data['prediction_date'] = target_date.strftime('%Y-%m-%d')
                
                if self.model_loaded and self.model is not None:
                    # ML 모델 예측 (4개 API 통합 특성 사용)
                    risk_score = self.predict_with_ml_model(data)
                else:
                    # 규칙 기반 예측
                    risk_score = self.calculate_simple_risk(data)
                
                risk_info = self.get_risk_level(risk_score)
                
                recommendations = {
                    0: ["정상적인 업무 진행", "일기예보 정기 확인"],
                    1: ["기상 상황 주시", "우산 준비"],
                    2: ["외출 시 주의", "지하공간 점검", "배수구 확인"],
                    3: ["불필요한 외출 자제", "중요 물품 안전한 곳 이동", "비상연락망 확인"],
                    4: ["즉시 대피 준비", "119 신고 대기", "지하시설 피해"]
                }
                
                return jsonify({
                    'risk_score': round(risk_score, 1),
                    'risk_level': risk_info['level'],
                    'risk_name': risk_info['name'],
                    'risk_color': risk_info['color'],
                    'action': risk_info['action'],
                    'recommendations': recommendations.get(risk_info['level'], []),
                    'prediction_time': datetime.now().isoformat(),
                    'prediction_date': data.get('prediction_date', datetime.now().strftime('%Y-%m-%d')),
                    'model_used': '4개 API 통합 ML Model' if self.model_loaded else 'Rule-based',
                    'data_freshness': (datetime.now() - self.data_last_updated).total_seconds() / 60 if self.data_last_updated else None,
                    'training_data_period': f"{self.data_start_date.date()} ~ {self.data_end_date.date()}" if self.data_start_date and self.data_end_date else None,
                    'feature_count': len(self.feature_names) if self.feature_names else 0
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def predict_with_ml_model(self, data):
        """ML 모델 예측 (4개 API 통합 특성 사용)"""
        try:
            features = []
            basic_features = {
                'precipitation': data.get('precipitation', 0),
                'humidity': data.get('humidity', 60),
                'avg_temp': data.get('avg_temp', 20),
                'wind_speed': data.get('wind_speed', 2),
                'month': datetime.now().month,
                'precip_ma3': data.get('precipitation', 0),
                'precip_ma7': data.get('precipitation', 0),
                'is_peak_rainy': 1 if data.get('season_type') == 'rainy' else 0,
                'precip_risk_level': self.get_precip_level(data.get('precipitation', 0)),
                'pressure': data.get('pressure', 1013),
                'data_quality_score': 100,  # 사용자 입력은 고품질로 가정
                'warning_risk_factor': 1.0,  # 기본값
                'active_warnings': 0  # 기본값
            }
            
            for feature_name in self.feature_names:
                if feature_name in basic_features:
                    features.append(basic_features[feature_name])
                else:
                    features.append(0)
            
            prediction_proba = self.model.predict_proba([features])[0][1]
            return prediction_proba * 100
            
        except Exception as e:
            return self.calculate_simple_risk(data)
    
    def get_precip_level(self, precipitation):
        """강수량 위험 등급"""
        if precipitation >= 100: return 4
        elif precipitation >= 50: return 3
        elif precipitation >= 30: return 2
        elif precipitation >= 10: return 1
        else: return 0
    
    def calculate_simple_risk(self, data):
        """간단한 위험도 계산 (4개 API 통합 고려)"""
        score = 0
        precipitation = data.get('precipitation', 0)
        score += min(precipitation * 0.4, 40)
        
        precip_3d = data.get('precip_sum_3d', precipitation)
        score += min(precip_3d * 0.25, 25)
        
        humidity = data.get('humidity', 50)
        score += min((humidity - 50) * 0.4, 20)
        
        season_type = data.get('season_type', 'dry')
        if season_type == 'rainy':
            score += 15
        else:
            score += 3
        
        # 기압 고려 (낮은 기압 = 위험 증가)
        pressure = data.get('pressure', 1013)
        if pressure < 1000:
            score += 10
        elif pressure < 1005:
            score += 5
        
        return min(score, 100)
    
    def get_risk_level(self, score):
        """위험도 등급"""
        if score <= 20:
            return {'level': 0, 'name': '매우낮음', 'color': '🟢', 'action': '정상 업무'}
        elif score <= 40:
            return {'level': 1, 'name': '낮음', 'color': '🟡', 'action': '상황 주시'}
        elif score <= 60:
            return {'level': 2, 'name': '보통', 'color': '🟠', 'action': '주의 준비'}
        elif score <= 80:
            return {'level': 3, 'name': '높음', 'color': '🔴', 'action': '대비 조치'}
        else:
            return {'level': 4, 'name': '매우높음', 'color': '🟣', 'action': '즉시 대응'}
    
    def run(self):
        """웹 서버 실행"""
        print("🇰🇷 침수 예측 AI 시스템 (4개 기상청 API 통합 + 3년 데이터)")
        print("📍 주소: http://localhost:5000")
        print("🗂️ 데이터 저장 경로:")
        print(f"   📁 원시 데이터: {self.raw_dir}/")
        print(f"   📁 처리된 데이터: {self.processed_dir}/")
        print(f"   📄 메인 파일: {self.processed_dir}/ML_COMPLETE_DATASET.csv")
        print("🆕 4개 API 통합 기능:")
        print("  - 📊 ASOS 시간자료 (가장 정확한 실시간 관측)")
        print("  - 📈 ASOS 일자료 (누적/통계 데이터)")
        print("  - 🚨 기상특보 (호우경보 등 침수 직접 경보)")
        print("  - 🌤️ 단기예보 (격자 기반 실황)")
        print("  - 📊 정확히 3년치 기상 데이터 자동 생성")
        print("  - ⏰ 1시간마다 자동 업데이트")
        print("  - 💾 자동 파일 저장")
        print("  - 📈 고품질 실시간 시각화")
        print("  - 🎯 기상특보 기반 위험도 가중치 적용")
        print("  - 🧠 4개 API 통합 특성으로 ML 모델 훈련")
        print("🛑 종료: Ctrl+C")
        
        self.app.run(debug=True, host='0.0.0.0', port=5000)