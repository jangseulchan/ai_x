#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_system.py - CREW_SOOM 시스템 상태 진단 스크립트
"""

import sys
import os
import importlib.util
from pathlib import Path

def print_banner():
    """진단 시작 배너"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🔍 CREW_SOOM 시스템 진단                           ║
║                    문제 해결을 위한 상태 점검 도구                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Python 버전 확인"""
    print("🐍 Python 환경 확인...")
    version = sys.version_info
    print(f"   Python 버전: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python 버전 요구사항 충족")
        return True
    else:
        print("   ❌ Python 3.8 이상이 필요합니다.")
        return False

def check_required_modules():
    """필수 모듈 확인"""
    print("\n📦 필수 모듈 확인...")
    
    required_modules = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'requests': 'requests',
        'flask': 'Flask'
    }
    
    missing_modules = []
    
    for module_name, package_name in required_modules.items():
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - 설치 필요")
            missing_modules.append(package_name)
    
    if missing_modules:
        print(f"\n💡 설치 명령: pip install {' '.join(missing_modules)}")
        return False
    
    return True

def check_optional_modules():
    """선택적 모듈 확인"""
    print("\n🔧 선택적 모듈 확인...")
    
    optional_modules = {
        'tensorflow': 'TensorFlow (딥러닝 모델용)',
        'xgboost': 'XGBoost (고성능 모델용)',
        'seaborn': 'Seaborn (시각화 향상)',
        'psutil': 'psutil (시스템 모니터링)'
    }
    
    for module_name, description in optional_modules.items():
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ⚠️ {description} - 선택사항")

def check_project_structure():
    """프로젝트 구조 확인"""
    print("\n📁 프로젝트 구조 확인...")
    
    required_files = [
        'modules/multi_weather_api.py',
        'modules/web_app.py',
        'modules/data_loader.py',
        'modules/preprocessor.py',
        'modules/trainer.py',
        'modules/evaluator.py',
        'modules/visualizer.py',
        'templates/dashboard.html',
        'templates/login.html',
        'static/css/style.css',
        'static/js/dashboard.js',
        'run.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - 파일 없음")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n💡 누락된 파일 {len(missing_files)}개를 확인하세요.")
        return False
    
    return True

def check_data_directories():
    """데이터 디렉토리 확인"""
    print("\n📊 데이터 디렉토리 확인...")
    
    required_dirs = [
        'data',
        'data/processed',
        'data/raw', 
        'models',
        'outputs',
        'logs'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ⚠️ {dir_path}/ - 자동 생성됨")
            os.makedirs(dir_path, exist_ok=True)

def check_env_file():
    """환경 설정 파일 확인"""
    print("\n🔑 환경 설정 확인...")
    
    if os.path.exists('.env'):
        print("   ✅ .env 파일 존재")
        
        # .env 파일 내용 확인
        try:
            with open('.env', 'r', encoding='utf-8') as f:
                content = f.read()
                if 'OPENWEATHER_API_KEY' in content:
                    print("   ✅ API 키 설정 확인됨")
                else:
                    print("   ⚠️ OPENWEATHER_API_KEY가 설정되지 않음")
        except Exception as e:
            print(f"   ⚠️ .env 파일 읽기 오류: {e}")
    else:
        print("   ❌ .env 파일 없음")
        print("   💡 .env 파일을 생성하고 API 키를 설정하세요:")
        print("      OPENWEATHER_API_KEY=your_api_key_here")

def test_module_imports():
    """모듈 import 테스트"""
    print("\n🧪 모듈 import 테스트...")
    
    test_modules = [
        ('modules.multi_weather_api', 'MultiWeatherAPI'),
        ('modules.web_app', 'AdvancedFloodWebApp'),
        ('modules.data_loader', 'DataLoader'),
        ('modules.trainer', 'AdvancedModelTrainer')
    ]
    
    for module_path, class_name in test_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                print(f"   ✅ {module_path}.{class_name}")
            else:
                print(f"   ❌ {module_path}.{class_name} - 클래스 없음")
        except ImportError as e:
            print(f"   ❌ {module_path} - Import 오류: {e}")
        except Exception as e:
            print(f"   ⚠️ {module_path} - 기타 오류: {e}")

def check_data_files():
    """데이터 파일 확인"""
    print("\n💾 데이터 파일 확인...")
    
    data_files = [
        ('data/processed/REAL_WEATHER_DATA.csv', '일자료'),
        ('data/processed/ASOS_HOURLY_DATA.csv', '시간자료')
    ]
    
    for file_path, description in data_files:
        if os.path.exists(file_path):
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                print(f"   ✅ {description}: {len(df)}행")
            except Exception as e:
                print(f"   ⚠️ {description}: 파일 손상됨 ({e})")
        else:
            print(f"   ❌ {description}: 파일 없음")

def suggest_fixes():
    """문제 해결 제안"""
    print("\n🛠️ 문제 해결 가이드:")
    print("   1. 필수 모듈 설치: pip install -r requirements.txt")
    print("   2. .env 파일 생성 및 API 키 설정")
    print("   3. 데이터 수집: python modules/multi_weather_api.py")
    print("   4. 웹 앱 실행: python run.py")
    print("\n📞 추가 도움이 필요하면:")
    print("   - GitHub Issues 확인")
    print("   - 로그 파일 확인: logs/ 디렉토리")
    print("   - 상세 오류: python run.py --verbose")

def main():
    """메인 진단 함수"""
    print_banner()
    
    checks = [
        ("Python 버전", check_python_version),
        ("필수 모듈", check_required_modules), 
        ("프로젝트 구조", check_project_structure),
        ("데이터 디렉토리", check_data_directories),
        ("환경 설정", check_env_file),
        ("모듈 Import", test_module_imports),
        ("데이터 파일", check_data_files)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ {check_name} 확인 중 오류: {e}")
            results.append((check_name, False))
    
    # 선택적 모듈 확인 (오류 무시)
    try:
        check_optional_modules()
    except:
        pass
    
    # 결과 요약
    print("\n" + "="*80)
    print("📋 진단 결과 요약:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"   {check_name}: {status}")
    
    print(f"\n🎯 전체 상태: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 확인 완료! 시스템이 정상 상태입니다.")
        print("💡 이제 python run.py로 시스템을 실행하세요.")
    else:
        print("⚠️ 일부 문제가 발견되었습니다.")
        suggest_fixes()
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 진단 중 예상치 못한 오류: {e}")
        sys.exit(1)