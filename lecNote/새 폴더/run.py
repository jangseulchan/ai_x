# run.py - CREW_SOOM 메인 실행 파일 (기존 구조 유지)
import os
import sys
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

# 필요한 디렉토리 생성
def ensure_directories():
    directories = [
        'static', 'static/css', 'static/js', 'static/images',
        'templates', 'modules', 'data', 'data/processed',
        'models', 'outputs', 'logs'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# 기본 CSS 파일이 없으면 생성
def create_default_css():
    css_path = 'static/css/style.css'
    if not os.path.exists(css_path):
        default_css = """
/* Elancer 스타일 기반 기본 CSS */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #4ECDC4;
    --success-color: #28a745;
    --warning-color: #ffc107;
    --danger-color: #dc3545;
    --white: #ffffff;
    --light-gray: #f8f9fa;
    --dark-gray: #343a40;
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: var(--dark-gray);
    background: var(--light-gray);
}

/* 기본 버튼 스타일 */
.btn {
    display: inline-block;
    padding: 12px 24px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 5px;
}

.btn-primary {
    background: var(--primary-gradient);
    color: var(--white);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

/* 로딩 표시 */
.loading {
    display: none;
    text-align: center;
    padding: 20px;
}

/* 반응형 */
@media (max-width: 768px) {
    .btn { font-size: 14px; padding: 10px 20px; }
}
"""
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(default_css)

# 기본 JS 파일이 없으면 생성
def create_default_js():
    js_path = 'static/js/dashboard.js'
    if not os.path.exists(js_path):
        default_js = """
// 기본 대시보드 JavaScript
console.log('CREW_SOOM Dashboard 로드됨');

// 상태 확인 함수
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        console.log('시스템 상태:', data);
        return data;
    } catch (error) {
        console.error('상태 확인 오류:', error);
        return null;
    }
}

// 페이지 로드시 실행
document.addEventListener('DOMContentLoaded', function() {
    console.log('페이지 로드 완료');
    checkStatus();
});
"""
        with open(js_path, 'w', encoding='utf-8') as f:
            f.write(default_js)

# 가상 데이터 생성 클래스
class DataSimulator:
    def __init__(self):
        self.data_count = 15420
        self.model_count = 4
        self.accuracy = 95.2
        self.last_update = datetime.now()
        
    def get_status(self):
        return {
            'data_loaded': True,
            'data_rows': self.data_count,
            'model_loaded': True,
            'models_count': self.model_count,
            'api_available': True,
            'today': datetime.now().strftime('%Y-%m-%d'),
            'accuracy': self.accuracy,
            'last_update': self.last_update.isoformat()
        }
    
    def predict_risk(self, input_data):
        # 간단한 위험도 계산
        precipitation = float(input_data.get('precipitation', 0))
        humidity = float(input_data.get('humidity', 60))
        
        score = min(100, precipitation * 0.8 + (humidity - 50) * 0.3)
        
        if score <= 20:
            level = {'level': 0, 'name': '매우낮음', 'color': '🟢', 'action': '정상 업무'}
        elif score <= 40:
            level = {'level': 1, 'name': '낮음', 'color': '🟡', 'action': '상황 주시'}
        elif score <= 60:
            level = {'level': 2, 'name': '보통', 'color': '🟠', 'action': '주의 준비'}
        elif score <= 80:
            level = {'level': 3, 'name': '높음', 'color': '🔴', 'action': '대비 조치'}
        else:
            level = {'level': 4, 'name': '매우높음', 'color': '🟣', 'action': '즉시 대응'}
        
        return {
            'success': True,
            'risk_score': score,
            'risk_level': level['level'],
            'risk_name': level['name'],
            'risk_color': level['color'],
            'action': level['action'],
            'prediction_time': datetime.now().isoformat(),
            'recommendations': [
                '기상 상황을 지속적으로 모니터링하세요',
                '우산을 준비하세요',
                '외출 시 주의하세요'
            ]
        }

# Flask 앱 생성
def create_app():
    app = Flask(__name__)
    app.secret_key = 'crew_soom_2024_secret_key'
    
    # 데이터 시뮬레이터
    data_sim = DataSimulator()
    
    # 라우트 설정
    @app.route('/')
    def index():
        return render_template('dashboard.html')
    
    @app.route('/dashboard')
    def dashboard():
        # 로그인된 사용자든 아니든 같은 페이지 표시 (로그인 상태는 JavaScript에서 체크)
        return render_template('dashboard.html')
    
    @app.route('/login')
    def login():
        return render_template('login.html')
    
    @app.route('/map')
    def map_page():
        return render_template('map.html')
    
    # API 라우트
    @app.route('/api/login', methods=['POST'])
    def api_login():
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # 간단한 로그인 (실제로는 데이터베이스 확인)
        if username == 'admin' and password == '1234':
            session['user'] = username
            return jsonify({'success': True, 'message': '로그인 성공'})
        else:
            return jsonify({'success': False, 'message': 'ID 또는 비밀번호가 틀립니다.'})
    
    @app.route('/api/logout')
    def api_logout():
        session.pop('user', None)
        return jsonify({'success': True})
    
    @app.route('/api/session')
    def api_session():
        return jsonify({'logged_in': 'user' in session})
    
    @app.route('/api/status')
    def api_status():
        return jsonify(data_sim.get_status())
    
    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        data = request.get_json()
        result = data_sim.predict_risk(data)
        return jsonify(result)
    
    @app.route('/api/chart/<chart_type>')
    def api_chart(chart_type):
        try:
            # 차트 생성
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == 'precipitation':
                # 강수량 차트
                dates = pd.date_range('2024-01-01', periods=30, freq='D')
                precip = np.random.exponential(5, 30)
                ax.plot(dates, precip, marker='o', alpha=0.7)
                ax.set_title('월별 강수량 추이')
                ax.set_ylabel('강수량 (mm)')
                
            elif chart_type == 'risk_distribution':
                # 위험도 분포
                risks = np.random.choice([0, 1, 2, 3, 4], 100, p=[0.4, 0.3, 0.2, 0.08, 0.02])
                risk_names = ['매우낮음', '낮음', '보통', '높음', '매우높음']
                colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336', '#9C27B0']
                
                unique, counts = np.unique(risks, return_counts=True)
                ax.bar([risk_names[i] for i in unique], counts, color=[colors[i] for i in unique])
                ax.set_title('위험도 분포')
                ax.set_ylabel('빈도')
                
            else:
                # 기본 차트
                x = np.linspace(0, 10, 100)
                y = np.sin(x)
                ax.plot(x, y)
                ax.set_title('기본 차트')
            
            plt.tight_layout()
            
            # 이미지를 base64로 변환
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_base64}'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    return app

if __name__ == '__main__':
    print("🌊 CREW_SOOM AI 침수 예측 플랫폼")
    print("=" * 50)
    
    # 디렉토리와 기본 파일 생성
    ensure_directories()
    create_default_css()
    create_default_js()
    
    try:
        # 기존 웹앱 모듈 import 시도
        from modules.web_app import AdvancedFloodWebApp
        print("✅ 고급 웹앱 모듈 로드 성공")
        
        # 웹앱 인스턴스 생성 및 실행
        app_instance = AdvancedFloodWebApp()
        app_instance.run()
        
    except ImportError as e:
        print(f"⚠️ 고급 모듈 로드 실패: {e}")
        print("📦 기본 모드로 실행합니다...")
        
        # 기본 Flask 앱으로 실행
        app = create_app()
        
        print("🚀 서버 시작 중...")
        print("📍 주소: http://localhost:5000")
        print("🔑 로그인: admin / 1234")
        print("🛑 종료: Ctrl+C")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        print("\n🔧 문제 해결 방법:")
        print("1. pip install -r requirements.txt")
        print("2. .env 파일에 API 키 설정")
        print("3. Python 버전 확인 (3.8 이상 필요)")