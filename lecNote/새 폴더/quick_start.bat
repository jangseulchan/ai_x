@echo off
title CREW_SOOM v2.0 고급 AI 침수 예측 시스템
color 0A

:menu
cls
echo.
echo  ████████████████████████████████████████████████████████████████████████████████
echo  ██                          🌊 CREW_SOOM v2.0                                 ██
echo  ██                     고급 AI 침수 예측 시스템                                ██
echo  ████████████████████████████████████████████████████████████████████████████████
echo.
echo  🤖 4가지 고급 AI 모델 지원:
echo     • RandomForest (앙상블)
echo     • XGBoost (그래디언트 부스팅)  
echo     • LSTM + CNN (하이브리드 딥러닝)
echo     • Transformer (어텐션 메커니즘)
echo.
echo  📋 메뉴를 선택하세요:
echo  1. 🔍 시스템 환경 체크
echo  2. 🚀 CREW_SOOM 시스템 실행
echo  3. 📊 테스트 데이터로 빠른 시작
echo  4. 🔧 패키지 재설치
echo  5. 📝 로그 파일 보기
echo  6. 🌐 웹 브라우저로 바로 열기
echo  0. ❌ 종료
echo.
set /p choice=선택 (0-6): 

if "%choice%"=="1" goto check
if "%choice%"=="2" goto run
if "%choice%"=="3" goto test
if "%choice%"=="4" goto reinstall
if "%choice%"=="5" goto logs
if "%choice%"=="6" goto browser
if "%choice%"=="0" goto exit
goto menu

:check
echo 🔍 시스템 환경 체크 중...
python check_system.py
pause
goto menu

:run
echo 🚀 CREW_SOOM 시스템 실행 중...
python run.py
pause
goto menu

:test
echo 📊 테스트 데이터로 빠른 시작...
set DEMO_MODE=True
python run.py
pause
goto menu

:reinstall
echo 🔧 패키지 재설치 중...
pip install -r requirements.txt
pause
goto menu

:logs
echo 📝 로그 파일 확인...
if exist logs\crew_soom.log (
    type logs\crew_soom.log
) else (
    echo 로그 파일이 없습니다.
)
pause
goto menu

:browser
echo 🌐 웹 브라우저 열기...
start http://localhost:5000
echo 💡 로그인 정보: admin / 1234
pause
goto menu

:exit
echo 👋 CREW_SOOM을 이용해 주셔서 감사합니다!
exit
