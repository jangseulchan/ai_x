@echo off
chcp 65001 > nul
echo  시스템 정리
echo =============
echo.
echo 임시 파일과 캐시를 정리합니다...
echo.
if exist __pycache__ rmdir /s /q __pycache__
if exist modules\__pycache__ rmdir /s /q modules\__pycache__
if exist *.pyc del *.pyc
if exist modules\*.pyc del modules\*.pyc
if exist logs\*.log del logs\*.log
echo.
echo 정리가 완료되었습니다!
pause