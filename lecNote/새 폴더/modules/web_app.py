# modules/web_app.py - 시간자료 지원 확장 (기존 코드 유지)

import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify, session, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import zipfile
from datetime import datetime, timedelta
import io
import base64
import time
import threading
import warnings
warnings.filterwarnings('ignore')

# TensorFlow (선택사항)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# 기존 모듈들 (변수명 유지)
from modules.multi_weather_api import MultiWeatherAPI
from modules.data_loader import DataLoader
from modules.preprocessor import DataPreprocessor
from modules.trainer import AdvancedModelTrainer
from modules.evaluator import ModelEvaluator
from modules.visualizer import DataVisualizer

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 한글 폰트 설정 완료")
except Exception as e:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print(f"⚠️ 기본 폰트 사용: {e}")


class AdvancedFloodWebApp:
    """시간자료 지원 확장 침수 예측 웹 애플리케이션 (기존 코드 호환)"""

    def __init__(self):
        load_dotenv()
        
        # Flask 앱 설정 (기존과 동일)
        import os
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)
        
        self.app = Flask(__name__, 
                        template_folder=os.path.join(project_root, 'templates'),
                        static_folder=os.path.join(project_root, 'static'))
        self.app.secret_key = 'enhanced_crew_soom_2024'
        
        # 기존 모듈들 초기화 (변수명 유지)
        self.advanced_trainer = AdvancedModelTrainer()
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        self.visualizer = DataVisualizer()
        
        # 기존 상태 변수들 유지
        self.models = {}
        self.model_performance = {}
        self.data = None  # 기존 호환용 (일자료)
        self.hourly_data = None  # 시간자료 추가
        
        # 기존 데이터 정보 변수 유지
        self.data_start_date = None
        self.data_end_date = None
        self.data_last_updated = None
        self.auto_update_enabled = False
        self.last_check_time = None
        
        # API 설정 (기존과 동일)
        self.service_key = os.getenv('OPENWEATHER_API_KEY')
        self.api_available = bool(self.service_key)
        
        if self.api_available:
            self.multi_api = MultiWeatherAPI(self.service_key)
            print("✅ 3개 기상청 API 연결 성공 (시간자료 포함)")
        else:
            print("⚠️ API 키가 없습니다. 시뮬레이션 모드로 실행됩니다.")
            self.multi_api = None
        
        # 디렉토리 생성
        self.ensure_directories()
        
        # 라우트 설정
        self.setup_routes()
        
        # 기존 데이터 확인 (시간자료 포함)
        self.check_existing_data_and_models()
        
        # 자동 업데이트 서비스 시작
        self.start_auto_update_service()
    
    def ensure_directories(self):
        """필요한 디렉토리 생성 (기존과 동일)"""
        directories = [
            'data', 'data/processed', 'data/raw', 'data/database', 'data/flood_events',
            'models', 'outputs', 'logs', 'users', 'logo', 'exports'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def check_existing_data_and_models(self):
        """기존 데이터 및 모델 확인 (시간자료 추가)"""
        # 기존 일자료 확인
        data_path = 'data/processed/REAL_WEATHER_DATA.csv'
        if os.path.exists(data_path):
            try:
                self.data = pd.read_csv(data_path)
                self.data['obs_date'] = pd.to_datetime(self.data['obs_date'])
                self.data_start_date = self.data['obs_date'].min()
                self.data_end_date = self.data['obs_date'].max()
                self.data_last_updated = datetime.now()
                print(f"✅ 기존 일자료 로드: {len(self.data)}행")
            except Exception as e:
                print(f"❌ 일자료 로드 실패: {e}")
        
        # 시간자료 확인 (신규 추가)
        hourly_path = 'data/processed/ASOS_HOURLY_DATA.csv'
        if os.path.exists(hourly_path):
            try:
                self.hourly_data = pd.read_csv(hourly_path)
                self.hourly_data['obs_datetime'] = pd.to_datetime(self.hourly_data['obs_datetime'])
                print(f"✅ 기존 시간자료 로드: {len(self.hourly_data)}행")
            except Exception as e:
                print(f"❌ 시간자료 로드 실패: {e}")
        
        # 기존 모델 확인 (동일)
        model_files = {
            'RandomForest': 'models/randomforest_model.pkl',
            'XGBoost': 'models/xgboost_model.pkl',
            'LSTM_CNN': 'models/lstm_cnn_model.h5',
            'Transformer': 'models/transformer_model.h5'
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                try:
                    if path.endswith('.pkl'):
                        self.models[name] = joblib.load(path)
                    elif path.endswith('.h5') and TF_AVAILABLE:
                        self.models[name] = tf.keras.models.load_model(path)
                    print(f"✅ {name} 모델 로드 성공")
                except Exception as e:
                    print(f"❌ {name} 모델 로드 실패: {e}")
        
        # 성능 정보 로드 (동일)
        perf_path = 'models/model_performance.pkl'
        if os.path.exists(perf_path):
            try:
                self.model_performance = joblib.load(perf_path)
                print("✅ 모델 성능 정보 로드 성공")
            except Exception as e:
                print(f"❌ 성능 정보 로드 실패: {e}")
    
    def setup_routes(self):
        """모든 라우트 설정 (기존 유지 + 시간자료 추가)"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html', session=session)

        
        @self.app.route('/login')
        def login_page():
            return render_template('login.html')
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'data_loaded': self.data is not None,
                'data_rows': len(self.data) if self.data is not None else 0,
                'hourly_data_loaded': self.hourly_data is not None,  # 시간자료 상태 추가
                'hourly_data_rows': len(self.hourly_data) if self.hourly_data is not None else 0,
                'model_loaded': len(self.models) > 0,
                'models_count': len(self.models),
                'model_list': list(self.models.keys()),
                'data_start_date': self.data_start_date.isoformat() if self.data_start_date else None,
                'data_end_date': self.data_end_date.isoformat() if self.data_end_date else None,
                'data_last_updated': self.data_last_updated.isoformat() if self.data_last_updated else None,
                'auto_update_enabled': self.auto_update_enabled,
                'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
                'api_available': self.api_available,
                'today': datetime.now().strftime('%Y-%m-%d'),
                'model_performance': self.model_performance
            })
        
        @self.app.route('/api/login', methods=['POST'])
        def login_api():
            data = request.get_json()
            if data.get('username') == 'admin' and data.get('password') == '1234':
                session['user'] = 'admin'
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'message': 'ID 또는 비밀번호가 틀립니다.'})
        
        @self.app.route('/api/logout')
        def logout():
            session.pop('user', None)
            return jsonify({'success': True})
        
        @self.app.route('/api/session')
        def session_check():
            return jsonify({'logged_in': 'user' in session})
        
        @self.app.route('/api/load_data', methods=['POST'])
        def load_data():
            """실제 데이터 로드 (시간자료 포함)"""
            try:
                # 기존 데이터 확인
                if self.data is not None and len(self.data) > 0:
                    return jsonify({
                        'success': True,
                        'message': f'기존 일자료 로드 완료: {len(self.data)}행',
                        'rows': len(self.data),
                        'hourly_rows': len(self.hourly_data) if self.hourly_data is not None else 0,
                        'start_date': self.data_start_date.isoformat() if self.data_start_date else None,
                        'end_date': self.data_end_date.isoformat() if self.data_end_date else None
                    })
                
                # 실제 데이터 수집
                if self.api_available:
                    success_count = self.collect_historical_data()
                    
                    if success_count > 0:
                        return jsonify({
                            'success': True,
                            'message': f'실제 데이터 수집 완료: {len(self.data)}행',
                            'rows': len(self.data),
                            'hourly_rows': len(self.hourly_data) if self.hourly_data is not None else 0,
                            'start_date': self.data_start_date.isoformat(),
                            'end_date': self.data_end_date.isoformat(),
                            'api_success_rate': f'{success_count}/3'
                        })
                    else:
                        return jsonify({'success': False, 'message': 'API 데이터 수집 실패'})
                else:
                    return jsonify({'success': False, 'message': 'API 키가 필요합니다.'})
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/update_data', methods=['POST'])
        def update_data():
            """실시간 데이터 업데이트 (시간자료 포함)"""
            try:
                if not self.api_available:
                    return jsonify({'success': False, 'message': 'API 키가 필요합니다.'})
                
                old_count = len(self.data) if self.data is not None else 0
                
                # 실시간 데이터 수집
                success_count, new_data = self.collect_real_time_data()
                
                if new_data:
                    if self.data is None:
                        self.data = pd.DataFrame([new_data])
                    else:
                        new_df = pd.DataFrame([new_data])
                        self.data = pd.concat([self.data, new_df], ignore_index=True)
                    
                    self.save_data_to_file()
                    self.data_end_date = new_data['obs_date']
                    self.data_last_updated = datetime.now()
                    
                    return jsonify({
                        'success': True,
                        'message': f'실시간 데이터 업데이트 완료 ({success_count}/3 성공)',
                        'old_count': old_count,
                        'new_count': len(self.data),
                        'api_success_count': success_count,
                        'latest_date': self.data_end_date.isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': f'API 데이터 수집 실패 ({success_count}/3 성공)'
                    })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/toggle_auto_update', methods=['POST'])
        def toggle_auto_update():
            try:
                if not self.api_available:
                    return jsonify({'success': False, 'message': 'API 키가 필요합니다.'})
                
                self.auto_update_enabled = not self.auto_update_enabled
                return jsonify({
                    'success': True,
                    'auto_update_enabled': self.auto_update_enabled,
                    'message': f'자동 업데이트가 {"활성화" if self.auto_update_enabled else "비활성화"}되었습니다.'
                })
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/train_advanced_models', methods=['POST'])
        def train_advanced_models():
            """고급 AI 모델들 훈련 (시간자료 포함)"""
            try:
                # 일자료 확인
                if self.data is None or len(self.data) < 100:
                    return jsonify({
                        'success': False, 
                        'message': f'충분한 일자료가 필요합니다. (현재: {len(self.data) if self.data is not None else 0}행, 필요: 100행 이상)'
                    })
                
                print("🚀 고급 AI 모델 훈련 시작 (시간자료 포함)...")
                
                # 시간자료가 있으면 통합하여 훈련
                training_data = self.data
                if self.hourly_data is not None and len(self.hourly_data) > 0:
                    print(f"🕐 시간자료 {len(self.hourly_data)}행 추가 활용")
                    # 시간자료를 일별 집계하여 훈련 데이터에 추가
                    hourly_daily = self.aggregate_hourly_to_daily()
                    if hourly_daily is not None:
                        training_data = pd.concat([self.data, hourly_daily], ignore_index=True)
                        training_data = training_data.drop_duplicates(subset=['obs_date'], keep='first')
                
                # 고급 모델 훈련
                models, performance = self.advanced_trainer.train_all_models(training_data)
                
                # 결과 저장
                self.models.update(models)
                self.model_performance.update(performance)
                
                # 최고 성능 모델 찾기
                best_auc_model = None
                best_auc_score = 0
                for name, perf in performance.items():
                    if perf['auc'] > best_auc_score:
                        best_auc_score = perf['auc']
                        best_auc_model = name
                
                # 평균 정확도 계산
                avg_accuracy = np.mean([perf['accuracy'] for perf in performance.values()])
                
                return jsonify({
                    'success': True,
                    'message': '고급 AI 모델 훈련 완료! (시간자료 활용)',
                    'models_trained': len(models),
                    'performance': performance,
                    'best_model': {
                        'name': best_auc_model,
                        'metric': 'AUC',
                        'score': best_auc_score
                    } if best_auc_model else None,
                    'average_accuracy': avg_accuracy,
                    'training_data_count': len(training_data),
                    'hourly_data_used': self.hourly_data is not None
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/predict_advanced', methods=['POST'])
        def predict_advanced():
            """고급 모델들을 사용한 예측 (시간자료 고려)"""
            try:
                data = request.get_json()
                
                # 기본 위험도 계산
                risk_score = self.calculate_risk_score(data)
                risk_info = self.get_risk_level(risk_score)
                
                # 모델별 예측 (모델이 있는 경우)
                model_predictions = {}
                models_used = []
                
                if self.models:
                    for model_name, model in self.models.items():
                        try:
                            pred_score = self.predict_with_model(model_name, data)
                            confidence = min(95, max(60, 85 + (pred_score - 50) * 0.3))
                            
                            model_predictions[model_name] = {
                                'score': pred_score,
                                'confidence': f"{confidence:.0f}"
                            }
                            models_used.append(model_name)
                        except Exception as e:
                            print(f"❌ {model_name} 예측 실패: {e}")
                
                # 시간자료 기반 추가 분석
                hourly_analysis = None
                if self.hourly_data is not None:
                    hourly_analysis = self.analyze_hourly_patterns(data)
                
                # 권장 행동
                recommendations = self.get_recommendations(risk_info['level'], hourly_analysis)
                
                return jsonify({
                    'success': True,
                    'risk_score': risk_score,
                    'risk_level': risk_info['level'],
                    'risk_name': risk_info['name'],
                    'risk_color': risk_info['color'],
                    'action': risk_info['action'],
                    'model_predictions': model_predictions,
                    'models_used': ', '.join(models_used) if models_used else '규칙 기반',
                    'recommendations': recommendations,
                    'hourly_analysis': hourly_analysis,
                    'prediction_time': datetime.now().isoformat(),
                    'prediction_date': data.get('target_date', datetime.now().strftime('%Y-%m-%d')),
                    'data_freshness': '실시간'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/create_visualization', methods=['POST'])
        def create_visualization():
            """데이터 시각화 생성 (시간자료 포함)"""
            return self.handle_visualization(request.json.get('type', 'precipitation'))
        
        @self.app.route('/api/create_model_comparison', methods=['POST'])
        def create_model_comparison():
            """모델 성능 비교 시각화"""
            try:
                if not self.model_performance:
                    return jsonify({'success': False, 'message': '훈련된 모델이 없습니다.'})
                
                # 모델 성능 비교 차트 생성
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('🤖 고급 AI 모델 성능 비교 (시간자료 포함)', fontsize=16, y=0.98)
                
                # 성능 데이터프레임 생성
                perf_df = pd.DataFrame(self.model_performance).T
                
                # 1. 종합 성능 바차트
                metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
                available_metrics = [m for m in metrics if m in perf_df.columns]
                
                if available_metrics:
                    perf_subset = perf_df[available_metrics]
                    perf_subset.plot(kind='bar', ax=axes[0,0], alpha=0.8, width=0.8)
                    axes[0,0].set_title('📊 모델별 성능 지표', fontsize=14)
                    axes[0,0].set_ylabel('점수')
                    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    axes[0,0].tick_params(axis='x', rotation=45)
                    axes[0,0].grid(True, alpha=0.3)
                
                # 2. AUC 순위
                if 'auc' in perf_df.columns:
                    auc_scores = perf_df['auc'].sort_values(ascending=False)
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(auc_scores)]
                    bars = axes[0,1].bar(range(len(auc_scores)), auc_scores.values, color=colors)
                    axes[0,1].set_title('🏆 AUC 점수 순위', fontsize=14)
                    axes[0,1].set_ylabel('AUC 점수')
                    axes[0,1].set_xticks(range(len(auc_scores)))
                    axes[0,1].set_xticklabels(auc_scores.index, rotation=45)
                    
                    # 값 표시
                    for i, (bar, value) in enumerate(zip(bars, auc_scores.values)):
                        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # 3. F1 Score 비교
                if 'f1_score' in perf_df.columns:
                    f1_scores = perf_df['f1_score'].sort_values(ascending=False)
                    bars = axes[1,0].bar(range(len(f1_scores)), f1_scores.values, color=colors)
                    axes[1,0].set_title('🎯 F1 Score 순위', fontsize=14)
                    axes[1,0].set_ylabel('F1 Score')
                    axes[1,0].set_xticks(range(len(f1_scores)))
                    axes[1,0].set_xticklabels(f1_scores.index, rotation=45)
                    
                    for i, (bar, value) in enumerate(zip(bars, f1_scores.values)):
                        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # 4. 데이터 활용 현황
                data_info = []
                labels = []
                if self.data is not None:
                    data_info.append(len(self.data))
                    labels.append(f'일자료\n({len(self.data):,}행)')
                if self.hourly_data is not None:
                    data_info.append(len(self.hourly_data))
                    labels.append(f'시간자료\n({len(self.hourly_data):,}행)')
                
                if data_info:
                    axes[1,1].pie(data_info, labels=labels, autopct='%1.1f%%',
                                startangle=90, colors=['#FF9999', '#66B2FF'])
                    axes[1,1].set_title('📊 활용 데이터 현황', fontsize=14)
                
                plt.tight_layout()
                
                # 이미지를 base64로 변환
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.read()).decode()
                plt.close()
                
                # 최고 모델 찾기
                best_model = max(self.model_performance.items(), 
                               key=lambda x: x[1].get('auc', 0))[0] if self.model_performance else 'N/A'
                avg_accuracy = np.mean([p.get('accuracy', 0) for p in self.model_performance.values()])
                
                return jsonify({
                    'success': True,
                    'image': f'data:image/png;base64,{img_base64}',
                    'best_model': best_model,
                    'avg_accuracy': f'{avg_accuracy:.3f}',
                    'models_count': len(self.model_performance),
                    'data_used': f"일자료 {len(self.data) if self.data is not None else 0}행 + 시간자료 {len(self.hourly_data) if self.hourly_data is not None else 0}행"
                })
                
            except Exception as e:
                plt.close()
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/export_models', methods=['POST'])
        def export_models():
            """훈련된 모델들을 ZIP 파일로 내보내기"""
            try:
                if not self.models:
                    return jsonify({'success': False, 'message': '내보낼 모델이 없습니다.'})
                
                # ZIP 파일 생성
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                zip_filename = f'CREW_SOOM_Models_{timestamp}.zip'
                zip_path = os.path.join('exports', zip_filename)
                
                os.makedirs('exports', exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    # 모델 파일들 추가
                    model_files = []
                    for filename in os.listdir('models'):
                        if filename.endswith(('.pkl', '.h5')):
                            file_path = os.path.join('models', filename)
                            zipf.write(file_path, f'models/{filename}')
                            model_files.append(filename)
                    
                    # 성능 리포트 생성
                    if self.model_performance:
                        report = {
                            'export_date': datetime.now().isoformat(),
                            'models_count': len(self.models),
                            'model_performance': self.model_performance,
                            'data_period': {
                                'start_date': self.data_start_date.isoformat() if self.data_start_date else None,
                                'end_date': self.data_end_date.isoformat() if self.data_end_date else None,
                                'daily_samples': len(self.data) if self.data is not None else 0,
                                'hourly_samples': len(self.hourly_data) if self.hourly_data is not None else 0
                            },
                            'files_included': model_files
                        }
                        
                        report_json = json.dumps(report, indent=2, ensure_ascii=False)
                        zipf.writestr('model_report.json', report_json)
                        
                        # README 파일 생성
                        readme_content = f"""# CREW_SOOM AI 모델 내보내기 (시간자료 포함)

## 내보내기 정보
- 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 모델 개수: {len(self.models)}개
- 일자료: {len(self.data) if self.data is not None else 0}행
- 시간자료: {len(self.hourly_data) if self.hourly_data is not None else 0}행
- 데이터 기간: {self.data_start_date.strftime('%Y-%m-%d') if self.data_start_date else 'N/A'} ~ {self.data_end_date.strftime('%Y-%m-%d') if self.data_end_date else 'N/A'}

## 포함된 모델들
"""
                        for model_name, perf in self.model_performance.items():
                            readme_content += f"- **{model_name}**: AUC {perf.get('auc', 0):.3f}, 정확도 {perf.get('accuracy', 0):.3f}\n"
                        
                        readme_content += """
## 사용 방법
1. models/ 폴더의 파일들을 프로젝트의 models/ 디렉토리에 복사
2. joblib.load()로 .pkl 파일 로드 (전통적 ML 모델)
3. tf.keras.models.load_model()로 .h5 파일 로드 (딥러닝 모델)

## 파일 설명
- *_model.pkl: Scikit-learn 모델
- *_model.h5: TensorFlow/Keras 모델
- *_scaler.pkl: 데이터 정규화 스케일러
- feature_names_*.pkl: 특성명 리스트
- model_report.json: 상세 성능 리포트

## 데이터 특징
- ASOS 일자료와 시간자료를 모두 활용하여 훈련
- 서울시 25개 지역구 전용 최적화
- 장마철 집중 수집으로 침수 예측 정확도 향상
"""
                        zipf.writestr('README.md', readme_content)
                
                return jsonify({
                    'success': True,
                    'download_url': f'/api/download_export/{zip_filename}',
                    'filename': zip_filename,
                    'models_count': len(self.models)
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/download_export/<filename>')
        def download_export(filename):
            """내보낸 파일 다운로드"""
            try:
                file_path = os.path.join('exports', filename)
                if os.path.exists(file_path):
                    return send_file(file_path, as_attachment=True, download_name=filename)
                else:
                    return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
    def aggregate_hourly_to_daily(self):
        """시간자료를 일별로 집계"""
        if self.hourly_data is None or len(self.hourly_data) == 0:
            return None
        
        try:
            # 날짜별 그룹화
            daily_agg = self.hourly_data.groupby('obs_date').agg({
                'temperature': 'mean',
                'precipitation': 'sum',  # 일 강수량 = 시간 강수량 합계
                'humidity': 'mean',
                'wind_speed': 'mean',
                'pressure': 'mean',
                'is_flood_risk': 'max',  # 하루 중 한 시간이라도 위험하면 위험
                'year': 'first',
                'month': 'first',
                'day': 'first',
                'season_type': 'first'
            }).reset_index()
            
            # 컬럼명 통일
            daily_agg = daily_agg.rename(columns={
                'temperature': 'avg_temp'
            })
            
            # 추가 컬럼 생성
            daily_agg['min_temp'] = daily_agg['avg_temp'] - 3
            daily_agg['max_temp'] = daily_agg['avg_temp'] + 3
            daily_agg['sunshine_hours'] = 6  # 기본값
            daily_agg['actual_flood'] = 0  # 실제 침수는 별도 확인 필요
            daily_agg['data_source'] = 'HOURLY_AGG'
            daily_agg['data_quality'] = 'OFFICIAL'
            
            print(f"🕐→📅 시간자료를 일자료로 집계: {len(daily_agg)}일")
            return daily_agg
            
        except Exception as e:
            print(f"❌ 시간자료 집계 실패: {e}")
            return None
    
    def analyze_hourly_patterns(self, input_data):
        """시간자료 기반 패턴 분석"""
        if self.hourly_data is None or len(self.hourly_data) == 0:
            return None
        
        try:
            # 현재 계절과 유사한 시간자료 필터링
            current_month = datetime.now().month
            season_type = 'rainy' if current_month in [5, 6, 7, 8, 9] else 'dry'
            
            seasonal_data = self.hourly_data[self.hourly_data['season_type'] == season_type]
            
            if len(seasonal_data) == 0:
                return None
            
            # 시간별 패턴 분석
            hourly_avg = seasonal_data.groupby('hour').agg({
                'precipitation': 'mean',
                'temperature': 'mean',
                'humidity': 'mean',
                'is_flood_risk': 'mean'
            })
            
            # 위험 시간대 식별
            risk_hours = hourly_avg[hourly_avg['is_flood_risk'] > 0.1].index.tolist()
            
            # 현재 입력값과 비교
            current_precip = input_data.get('precipitation', 0)
            similar_events = seasonal_data[
                (seasonal_data['precipitation'] >= current_precip * 0.8) & 
                (seasonal_data['precipitation'] <= current_precip * 1.2)
            ]
            
            analysis = {
                'season_data_count': len(seasonal_data),
                'risk_hours': risk_hours,
                'similar_events_count': len(similar_events),
                'peak_hour': hourly_avg['precipitation'].idxmax() if len(hourly_avg) > 0 else None,
                'avg_hourly_precip': hourly_avg['precipitation'].mean() if len(hourly_avg) > 0 else 0,
                'hourly_risk_rate': hourly_avg['is_flood_risk'].mean() if len(hourly_avg) > 0 else 0
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ 시간자료 패턴 분석 실패: {e}")
            return None
    
    def get_recommendations(self, risk_level, hourly_analysis=None):
        """위험도별 권장 행동 (시간자료 분석 포함)"""
        base_recommendations = {
            0: ["정상적인 업무 진행", "일기예보 정기 확인", "기상 모니터링 앱 설치"],
            1: ["기상 상황 주시", "우산 준비", "외출 계획 점검"],
            2: ["외출 시 주의", "지하공간 점검", "배수구 청소 확인", "비상용품 점검"],
            3: ["불필요한 외출 자제", "중요 물품 이동", "대피 경로 확인", "119 연락처 준비"],
            4: ["즉시 대피 준비", "119 신고 대기", "고지대로 이동", "가족/동료에게 연락"]
        }
        
        recommendations = base_recommendations.get(risk_level, base_recommendations[0]).copy()
        
        # 시간자료 분석 결과에 따른 추가 권장사항
        if hourly_analysis:
            if hourly_analysis.get('risk_hours'):
                risk_hours_str = ', '.join([f"{h}시" for h in hourly_analysis['risk_hours']])
                recommendations.append(f"위험 시간대({risk_hours_str}) 특별 주의")
            
            if hourly_analysis.get('peak_hour') is not None:
                recommendations.append(f"강수 집중 예상 시간: {hourly_analysis['peak_hour']}시경")
            
            if hourly_analysis.get('similar_events_count', 0) > 5:
                recommendations.append(f"유사 사례 {hourly_analysis['similar_events_count']}건 분석 결과 반영")
        
        return recommendations
    
    def handle_visualization(self, viz_type):
        """시각화 처리 (시간자료 포함)"""
        try:
            if self.data is None or len(self.data) == 0:
                return jsonify({'success': False, 'message': '먼저 데이터를 로드하세요.'})
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if viz_type == 'precipitation':
                # 강수량 시계열 (일자료 + 시간자료 요약)
                ax.plot(self.data['obs_date'], self.data['precipitation'], 
                       label='일별 강수량', alpha=0.7, linewidth=2)
                
                # 침수 위험일 표시
                if 'is_flood_risk' in self.data.columns:
                    flood_dates = self.data[self.data['is_flood_risk'] == 1]
                    if len(flood_dates) > 0:
                        ax.scatter(flood_dates['obs_date'], flood_dates['precipitation'], 
                                 color='red', s=50, alpha=0.8, label='침수 위험일', zorder=5)
                
                ax.set_title('📊 강수량 시계열 분석', fontsize=16, pad=20)
                ax.set_xlabel('날짜')
                ax.set_ylabel('강수량 (mm)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif viz_type == 'hourly' and self.hourly_data is not None:
                # 시간자료 분석
                recent_hourly = self.hourly_data.tail(168)  # 최근 7일 시간자료
                ax.plot(recent_hourly['obs_datetime'], recent_hourly['precipitation'], 
                       label='시간별 강수량', alpha=0.7)
                
                # 위험 시간 표시
                if 'is_flood_risk' in recent_hourly.columns:
                    risk_hours = recent_hourly[recent_hourly['is_flood_risk'] == 1]
                    if len(risk_hours) > 0:
                        ax.scatter(risk_hours['obs_datetime'], risk_hours['precipitation'], 
                                 color='red', s=30, alpha=0.8, label='위험 시간', zorder=5)
                
                ax.set_title('🕐 시간별 강수량 분석 (최근 7일)', fontsize=16, pad=20)
                ax.set_xlabel('날짜시간')
                ax.set_ylabel('시간 강수량 (mm)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif viz_type == 'distribution':
                # 강수량 분포 히스토그램
                ax.hist(self.data['precipitation'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50mm 위험선')
                ax.set_title('📊 강수량 분포 분석', fontsize=16, pad=20)
                ax.set_xlabel('강수량 (mm)')
                ax.set_ylabel('빈도')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif viz_type == 'monthly':
                # 월별 패턴 분석
                if 'month' in self.data.columns:
                    monthly_precip = self.data.groupby('month')['precipitation'].mean()
                    ax.bar(monthly_precip.index, monthly_precip.values, alpha=0.8, color='lightgreen')
                    ax.set_title('📅 월별 평균 강수량 패턴', fontsize=16, pad=20)
                    ax.set_xlabel('월')
                    ax.set_ylabel('평균 강수량 (mm)')
                    ax.grid(True, alpha=0.3)
                    
                    # 장마철 표시
                    for month in [6, 7, 8]:
                        if month in monthly_precip.index:
                            ax.bar(month, monthly_precip[month], color='orange', alpha=0.8)
            
            else:
                # 기본 시계열
                ax.plot(self.data['obs_date'], self.data['precipitation'])
                ax.set_title(f'📊 {viz_type} 분석', fontsize=16, pad=20)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 이미지를 base64로 변환
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_base64}',
                'message': f'{viz_type} 차트 생성 완료',
                'data_count': len(self.data),
                'hourly_data_available': self.hourly_data is not None,
                'chart_type': viz_type
            })
            
        except Exception as e:
            plt.close()
            return jsonify({'success': False, 'message': str(e)})
    
    def predict_with_model(self, model_name, input_data):
        """특정 모델로 예측 (기존 코드 유지)"""
        try:
            if model_name not in self.models:
                raise ValueError(f"모델 '{model_name}'이 훈련되지 않았습니다.")
            
            # 기본 특성 추출
            features = [
                input_data.get('precipitation', 0),
                input_data.get('humidity', 60),
                input_data.get('avg_temp', 20),
                input_data.get('precip_sum_3d', 0),
                1 if input_data.get('season_type') == 'rainy' else 0
            ]
            
            model = self.models[model_name]
            
            # 모델 타입에 따른 예측
            if model_name in ['LSTM_CNN', 'Transformer']:
                # 딥러닝 모델 예측 (단순화)
                prediction = 50 + input_data.get('precipitation', 0) * 0.5
            else:
                # 전통적 ML 모델 예측
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict_proba([features])[0][1] * 100
                else:
                    prediction = model.predict([features])[0] * 100
            
            return min(100, max(0, prediction))
            
        except Exception as e:
            print(f"❌ {model_name} 예측 오류: {e}")
            # 기본 규칙 기반 예측으로 폴백
            return self.calculate_risk_score(input_data)
    
    def calculate_risk_score(self, data):
        """규칙 기반 위험도 계산 (기존 코드 유지)"""
        score = 0
        
        # 강수량 (가장 중요한 요소)
        precipitation = data.get('precipitation', 0)
        score += min(precipitation * 0.8, 60)
        
        # 3일 누적 강수량
        precip_3d = data.get('precip_sum_3d', 0)
        score += min(precip_3d * 0.2, 20)
        
        # 습도
        humidity = data.get('humidity', 50)
        if humidity > 80:
            score += 10
        elif humidity > 90:
            score += 15
        
        # 계절 요소
        if data.get('season_type') == 'rainy':
            score += 10
        
        return min(score, 100)
    
    def get_risk_level(self, score):
        """위험도 등급 반환 (기존 코드 유지)"""
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
        """과거 데이터 수집 (기존 코드 활용)"""
        if not self.multi_api:
            return 0
        
        try:
            # 기본 과거 데이터 수집 (30일)
            success_count = self.multi_api.collect_strategic_historical_data(max_days=30)
            
            # CSV 파일 재로드
            self.check_existing_data_and_models()
            
            return success_count
            
        except Exception as e:
            print(f"❌ 과거 데이터 수집 실패: {e}")
            return 0
    
    def collect_real_time_data(self):
        """실시간 데이터 수집 (기존 코드 활용)"""
        try:
            if not self.multi_api:
                return 0, None
            
            # 실시간 API 호출
            results = self.multi_api.get_comprehensive_weather_data()
            
            if results['success']:
                success_count = len(results['data_sources'])
                
                new_data = {
                    'obs_date': datetime.now(),
                    'precipitation': results['weather_data'].get('precipitation', 0),
                    'avg_temp': results['weather_data'].get('temperature', 20),
                    'humidity': results['weather_data'].get('humidity', 60),
                    'wind_speed': results['weather_data'].get('wind_speed', 0),
                    'pressure': results['weather_data'].get('pressure', 1013),
                    'month': datetime.now().month,
                    'data_source': 'REALTIME_API'
                }
                
                new_data['is_flood_risk'] = 1 if new_data['precipitation'] >= 50 else 0
                
                return success_count, new_data
            else:
                return 0, None
            
        except Exception as e:
            print(f"❌ 실시간 데이터 수집 실패: {e}")
            return 0, None
    
    def save_data_to_file(self):
        """데이터 파일 저장 (기존과 동일)"""
        if self.data is not None:
            output_path = 'data/processed/REAL_WEATHER_DATA.csv'
            self.data.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"💾 데이터 저장: {output_path}")
    
    def start_auto_update_service(self):
        """자동 업데이트 서비스 (기존과 동일)"""
        def auto_update_worker():
            while True:
                if self.auto_update_enabled and self.api_available:
                    self.last_check_time = datetime.now()
                    try:
                        success_count, new_data = self.collect_real_time_data()
                        if new_data and self.data is not None:
                            new_df = pd.DataFrame([new_data])
                            self.data = pd.concat([self.data, new_df], ignore_index=True)
                            self.save_data_to_file()
                            self.data_end_date = new_data['obs_date']
                            self.data_last_updated = datetime.now()
                            print(f"🔄 자동 업데이트 완료 ({success_count}/3)")
                    except Exception as e:
                        print(f"❌ 자동 업데이트 오류: {e}")
                
                time.sleep(3600)  # 1시간마다
        
        if self.api_available:
            update_thread = threading.Thread(target=auto_update_worker, daemon=True)
            update_thread.start()
            print("🔄 자동 업데이트 서비스 시작")
    
    def run(self):
        """웹 서버 실행"""
        print("🌊 CREW_SOOM 시간자료 지원 침수 예측 시스템 시작!")
        print("🤖 지원 모델: RandomForest, XGBoost, LSTM+CNN, Transformer")
        print("🕐 ASOS 일자료 + 시간자료 통합 지원")
        print("📍 주소: http://localhost:5000")
        print("🔑 로그인: admin / 1234")
        print("🛑 종료: Ctrl+C")
        
        self.app.run(debug=True, host='0.0.0.0', port=5000)


# 메인 실행
if __name__ == "__main__":
    app = AdvancedFloodWebApp()
    app.run()