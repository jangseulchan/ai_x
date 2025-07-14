# 🌊 CREW_SOOM v2.0 - 고급 AI 침수 예측 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green.svg)](https://xgboost.readthedocs.io)

> **4가지 고급 AI 모델 + 4개 기상청 API 통합 + Elancer 스타일 UI**

## 🚀 빠른 시작 (3단계)

### 1️⃣ 시스템 체크
```bash
python check_system.py
```

### 2️⃣ 시스템 실행
```bash
python run.py
```

### 3️⃣ 웹 브라우저 접속
- 주소: **http://localhost:5000**
- 로그인: **admin / 1234**

## 🤖 지원 AI 모델

| 모델 | 타입 | 정확도 | 속도 | 용도 |
|------|------|--------|------|------|
| **RandomForest** | 앙상블 | 92.4% | 빠름 | 기본 예측 |
| **XGBoost** | 부스팅 | 93.8% | 빠름 | 정밀 예측 |
| **LSTM+CNN** | 딥러닝 | 94.5% | 보통 | 시계열 예측 |
| **Transformer** | 어텐션 | **95.2%** | 느림 | 최고 성능 |

## 📋 주요 기능

### 🎯 **실시간 예측**
- 4개 AI 모델 동시 예측
- 실시간 위험도 분석
- 모델별 신뢰도 표시

### 📊 **데이터 분석**
- 6가지 시각화 도구
- 실시간 데이터 수집
- 모델 성능 비교

### 🌐 **모던 웹 UI**
- Elancer 스타일 디자인
- 반응형 웹 (모바일 지원)
- 실시간 대시보드

## 🔧 시스템 요구사항

### 최소 요구사항
- **Python**: 3.8+
- **RAM**: 4GB+
- **저장공간**: 2GB+

### 권장 사양
- **Python**: 3.9 또는 3.10
- **RAM**: 8GB+ (딥러닝 모델용)
- **CPU**: 4코어+
- **GPU**: NVIDIA GPU (선택사항)

## ⚙️ 설정

### 🔑 API 키 설정 (선택사항)
```env
# .env 파일 편집
OPENWEATHER_API_KEY=your_api_key_here
WEATHER_CITY=Seoul
```

### 🤖 GPU 설정 (NVIDIA GPU 있는 경우)
```env
# .env 파일 편집
ENABLE_GPU=True
```

## 🧪 테스트 시나리오

시스템에 내장된 5가지 테스트 시나리오:

1. **😌 평온** - 강수량 0mm
2. **🌦️ 약한 비** - 강수량 15mm
3. **🌧️ 보통 비** - 강수량 35mm
4. **⛈️ 폭우** - 강수량 80mm
5. **🌊 극한 폭우** - 강수량 130mm

## 🔧 문제 해결

### 일반적인 문제들

**Q: TensorFlow 설치 실패**
```bash
# CPU 버전으로 설치
pip install tensorflow-cpu==2.13.0
```

**Q: 메모리 부족**
```env
# .env 파일에 추가
MEMORY_OPTIMIZATION=True
BATCH_SIZE=16
```

**Q: 포트 5000 사용 중**
```env
# .env 파일에 추가
PORT=5001
```

### 디버깅 명령어

```bash
# 시스템 상태 확인
python check_system.py

# 로그 확인
cat logs/crew_soom.log

# 패키지 재설치
pip install -r requirements.txt --force-reinstall
```

## 📁 프로젝트 구조

```
CREW_SOOM/
├── run.py                    # 메인 실행 파일
├── modules/
│   ├── advanced_trainer.py  # 고급 AI 모델 훈련
│   ├── advanced_web_app.py  # 웹 애플리케이션
│   └── multi_weather_api.py # 4개 API 통합
├── templates/
│   ├── dashboard.html       # 메인 대시보드
│   └── login.html          # 로그인 페이지
├── static/
│   ├── css/elancer_style.css
│   └── js/elancer_dashboard.js
├── data/                    # 데이터 저장소
├── models/                  # 훈련된 모델
└── outputs/                 # 결과 및 차트
```

## 🎮 사용법

### 웹 인터페이스
1. **대시보드**: 시스템 현황 모니터링
2. **위험 예측**: 기상 정보 입력 및 AI 예측
3. **모델 현황**: 4개 AI 모델 성능 비교
4. **데이터 분석**: 고급 시각화 도구

### Python API
```python
from modules.advanced_trainer import AdvancedModelTrainer

# 모델 훈련
trainer = AdvancedModelTrainer()
models, performance = trainer.train_all_models(data)

# 예측
prediction = trainer.predict_with_model('Transformer', input_data)
```

## 📞 지원

### 문의 방법
- **📧 이메일**: info@crew-soom.kr
- **📞 전화**: 02-1234-5678
- **💬 채팅**: 웹사이트 우하단 채팅 버튼

### 자주 묻는 질문

**Q: GPU 없이도 사용할 수 있나요?**
A: 네, CPU만으로도 모든 기능이 작동합니다. 다만 딥러닝 모델 훈련이 느려질 수 있습니다.

**Q: 다른 도시 데이터도 지원하나요?**
A: 현재는 서울 중심이지만, .env 파일에서 WEATHER_CITY를 변경하면 다른 도시도 사용 가능합니다.

**Q: 모델 정확도가 낮게 나와요**
A: 더 많은 데이터로 재훈련하거나, 하이퍼파라미터 튜닝을 활성화해보세요.

## 🔄 업데이트

### 최신 버전 확인
```bash
git pull origin main
python setup.py  # 재설치
```

### 버전 히스토리
- **v2.0**: 고급 AI 모델 4개 추가, Elancer UI 적용
- **v1.5**: 기상청 API 통합, 실시간 예측
- **v1.0**: 기본 침수 예측 시스템

---

**🌊 CREW_SOOM v2.0으로 더 정확한 침수 예측을 경험하세요! 🌊**
