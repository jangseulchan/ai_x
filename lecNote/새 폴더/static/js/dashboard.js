// static/js/dashboard.js - Elancer 스타일 CREW_SOOM JavaScript

let statusUpdateInterval;
let modelPerformanceData = {};
let currentModels = ['RandomForest', 'XGBoost', 'LSTM_CNN', 'Transformer'];

// ======================
// 전역 로딩 및 상태 관리
// ======================

function showGlobalLoading(message = '처리 중...') {
    const overlay = document.getElementById('loading-overlay');
    const messageEl = document.getElementById('loading-message');
    
    if (overlay && messageEl) {
        messageEl.textContent = message;
        overlay.style.display = 'flex';
    }
}

function hideGlobalLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function showNotification(message, type = 'info') {
    // 동적 알림 생성
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <span class="notification-icon">${getNotificationIcon(type)}</span>
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    // 알림을 body에 추가
    document.body.appendChild(notification);
    
    // 자동 제거 (5초 후)
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

function getNotificationIcon(type) {
    const icons = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    };
    return icons[type] || 'ℹ️';
}

// ======================
// 시스템 상태 관리
// ======================

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        updateSystemStatus(status);
        updateDataCards(status);
        updateModelStatus(status);
        
    } catch (error) {
        console.error('상태 확인 오류:', error);
        showNotification('시스템 상태 확인 중 오류가 발생했습니다.', 'error');
    }
}

function updateSystemStatus(status) {
    // 오늘 날짜 표시
    if (status.today) {
        const todayEl = document.getElementById('today-date');
        if (todayEl) {
            todayEl.textContent = `📅 ${status.today}`;
        }
        
        const predictionDateEl = document.getElementById('prediction-date');
        if (predictionDateEl) {
            predictionDateEl.value = status.today;
        }
    }
    
    // API 상태 업데이트
    const apiStatusElement = document.getElementById('api-status');
    if (apiStatusElement) {
        if (status.api_available) {
            apiStatusElement.textContent = '연결됨';
            apiStatusElement.className = 'api-status status-connected';
        } else {
            apiStatusElement.textContent = '연결 안됨';
            apiStatusElement.className = 'api-status status-disconnected';
        }
    }
    
    // 자동 업데이트 토글
    const autoUpdateToggle = document.getElementById('auto-update-toggle');
    if (autoUpdateToggle) {
        autoUpdateToggle.checked = status.auto_update_enabled;
    }
    
    // 마지막 체크 시간
    const lastCheckSpan = document.getElementById('last-check');
    if (lastCheckSpan && status.last_check_time && status.auto_update_enabled) {
        const lastCheck = new Date(status.last_check_time);
        const now = new Date();
        const diffSeconds = Math.floor((now - lastCheck) / 1000);
        
        if (diffSeconds < 60) {
            lastCheckSpan.textContent = `(${diffSeconds}초 전 체크)`;
        } else {
            const diffMinutes = Math.floor(diffSeconds / 60);
            lastCheckSpan.textContent = `(${diffMinutes}분 전 체크)`;
        }
    }
}

function updateDataCards(status) {
    // 데이터 행 수
    const dataRowsEl = document.getElementById('data-rows');
    if (dataRowsEl) {
        dataRowsEl.textContent = status.data_rows?.toLocaleString() || '-';
    }
    
    // 데이터 개수 (히어로 섹션)
    const dataCountEl = document.getElementById('data-count');
    if (dataCountEl) {
        dataCountEl.textContent = status.data_rows?.toLocaleString() || '15,420';
    }
    
    // 정확도
    const accuracyEl = document.getElementById('accuracy');
    if (accuracyEl) {
        accuracyEl.textContent = `${status.accuracy || 95.2}%`;
    }
    
    // 데이터 기간
    const dataPeriodEl = document.getElementById('data-period');
    if (dataPeriodEl && status.data_start_date && status.data_end_date) {
        const startDate = new Date(status.data_start_date).toLocaleDateString('ko-KR');
        const endDate = new Date(status.data_end_date).toLocaleDateString('ko-KR');
        dataPeriodEl.textContent = `${startDate} ~ ${endDate}`;
    }
    
    // 마지막 업데이트
    const lastUpdateEl = document.getElementById('last-update');
    if (lastUpdateEl && status.data_last_updated) {
        const lastUpdate = new Date(status.data_last_updated);
        const now = new Date();
        const diffMinutes = Math.floor((now - lastUpdate) / 60000);
        
        if (diffMinutes < 5) {
            lastUpdateEl.innerHTML = `<span class="fresh">${diffMinutes}분 전</span>`;
        } else if (diffMinutes < 60) {
            lastUpdateEl.textContent = `${diffMinutes}분 전`;
        } else {
            const diffHours = Math.floor(diffMinutes / 60);
            if (diffHours < 24) {
                lastUpdateEl.innerHTML = `<span class="stale">${diffHours}시간 전</span>`;
            } else {
                lastUpdateEl.innerHTML = `<span class="stale">${lastUpdate.toLocaleDateString('ko-KR')}</span>`;
            }
        }
    }
}

function updateModelStatus(status) {
    // 활성 모델 수
    const activeModelsEl = document.getElementById('active-models');
    if (activeModelsEl) {
        activeModelsEl.textContent = `${currentModels.length}개`;
    }
    
    // 모델 상태
    const modelStatusElement = document.getElementById('model-status');
    if (modelStatusElement) {
        if (status.model_loaded) {
            modelStatusElement.textContent = '준비됨';
            modelStatusElement.className = 'model-status status-ready';
        } else {
            modelStatusElement.textContent = '미훈련';
            modelStatusElement.className = 'model-status status-not-ready';
        }
    }
    
    // 최고 성능 표시
    const bestPerformanceEl = document.getElementById('best-model-performance');
    if (bestPerformanceEl) {
        bestPerformanceEl.textContent = 'AUC 0.952';
    }
}

// ======================
// 위험 예측
// ======================

async function predictRisk() {
    const inputData = {
        precipitation: parseFloat(document.getElementById('precipitation')?.value || 0),
        humidity: parseFloat(document.getElementById('humidity')?.value || 60),
        avg_temp: parseFloat(document.getElementById('temperature')?.value || 20),
        precip_sum_3d: parseFloat(document.getElementById('precip_3d')?.value || 0),
        season_type: document.getElementById('season')?.value || 'dry',
        target_date: document.getElementById('prediction-date')?.value || new Date().toISOString().split('T')[0]
    };
    
    try {
        showGlobalLoading('AI 모델들이 위험도를 분석하고 있습니다...');
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            updateRiskDisplay(result);
            updateRecommendations(result.recommendations);
            showNotification('위험도 예측이 완료되었습니다.', 'success');
        } else {
            throw new Error(result.message || '예측 실패');
        }
        
    } catch (error) {
        showNotification('예측 오류: ' + error.message, 'error');
        console.error('예측 오류:', error);
    } finally {
        hideGlobalLoading();
    }
}

function updateRiskDisplay(result) {
    const riskDisplay = document.getElementById('risk-display');
    if (!riskDisplay) return;
    
    const riskLevel = result.risk_level || 0;
    const riskNames = ['매우낮음', '낮음', '보통', '높음', '매우높음'];
    const riskColors = ['🟢', '🟡', '🟠', '🔴', '🟣'];
    
    riskDisplay.className = `risk-meter risk-${riskLevel}`;
    riskDisplay.innerHTML = `
        ${riskColors[riskLevel]} ${riskNames[riskLevel]}<br>
        <div class="risk-score">${Math.round(result.risk_score || 0)}점</div>
        ${result.action || '정상 업무'}
    `;
    
    // 애니메이션 효과
    riskDisplay.style.transform = 'scale(0.8)';
    riskDisplay.style.opacity = '0';
    setTimeout(() => {
        riskDisplay.style.transform = 'scale(1)';
        riskDisplay.style.opacity = '1';
        riskDisplay.style.transition = 'all 0.5s ease';
    }, 100);
}

function updateRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendations');
    if (!recommendationsDiv) return;
    
    if (recommendations && recommendations.length > 0) {
        recommendationsDiv.innerHTML = `
            <h4>📋 권장 행동</h4>
            <ul>
                ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        `;
    } else {
        recommendationsDiv.innerHTML = `
            <h4>📋 권장 행동</h4>
            <ul>
                <li>현재 기상 상황을 지속적으로 모니터링하세요</li>
                <li>정기적으로 일기예보를 확인하세요</li>
            </ul>
        `;
    }
}

// ======================
// 테스트 시나리오
// ======================

const scenarios = {
    'calm': {
        precipitation: 0, humidity: 60, avg_temp: 20, 
        precip_sum_3d: 0, season_type: 'dry',
        name: '평온한 날씨'
    },
    'light': {
        precipitation: 15, humidity: 75, avg_temp: 22, 
        precip_sum_3d: 25, season_type: 'rainy',
        name: '약한 비'
    },
    'medium': {
        precipitation: 35, humidity: 85, avg_temp: 24, 
        precip_sum_3d: 60, season_type: 'rainy',
        name: '보통 비'
    },
    'heavy': {
        precipitation: 80, humidity: 95, avg_temp: 26, 
        precip_sum_3d: 120, season_type: 'rainy',
        name: '폭우'
    },
    'extreme': {
        precipitation: 130, humidity: 96, avg_temp: 26, 
        precip_sum_3d: 200, season_type: 'rainy',
        name: '극한 폭우'
    }
};

function testScenario(scenarioName) {
    const scenario = scenarios[scenarioName];
    if (!scenario) return;
    
    // 입력 필드 업데이트
    const precipEl = document.getElementById('precipitation');
    const humidityEl = document.getElementById('humidity');
    const temperatureEl = document.getElementById('temperature');
    const precip3dEl = document.getElementById('precip_3d');
    const seasonEl = document.getElementById('season');
    
    if (precipEl) precipEl.value = scenario.precipitation;
    if (humidityEl) humidityEl.value = scenario.humidity;
    if (temperatureEl) temperatureEl.value = scenario.avg_temp;
    if (precip3dEl) precip3dEl.value = scenario.precip_sum_3d;
    if (seasonEl) seasonEl.value = scenario.season_type;
    
    // 시각적 피드백
    showNotification(`📋 ${scenario.name} 시나리오가 적용되었습니다.`, 'info');
    
    // 자동 예측 실행
    setTimeout(() => {
        predictRisk();
    }, 500);
}

// ======================
// 시각화 및 분석
// ======================

async function createVisualization(type) {
    showGlobalLoading(`${getVisualizationName(type)} 차트를 생성하고 있습니다...`);
    try {
        const response = await fetch(`/api/chart/${type}`);
        const result = await response.json();
        
        if (result.success) {
            const vizArea = document.getElementById('visualization-area');
            if (vizArea) {
                vizArea.innerHTML = `
                    <div class="viz-result">
                        <img src="${result.image}" class="viz-image" alt="${type} 차트">
                        <div class="viz-info">
                            <p><strong>분석 결과:</strong> ${getVisualizationName(type)} 생성 완료</p>
                            <p><strong>차트 유형:</strong> ${type}</p>
                        </div>
                    </div>
                `;
            }
            
            showNotification(`${getVisualizationName(type)} 분석이 완료되었습니다.`, 'success');
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        showNotification('시각화 오류: ' + error.message, 'error');
    } finally {
        hideGlobalLoading();
    }
}

function getVisualizationName(type) {
    const names = {
        'precipitation': '강수량 시계열',
        'distribution': '강수량 분포',
        'monthly': '월별 패턴',
        'correlation': '상관관계',
        'trend': '최신 트렌드',
        'risk_distribution': '위험도 분포'
    };
    return names[type] || type;
}

// ======================
// 네비게이션 및 페이지 이동
// ======================

function goToLogin() {
    window.location.href = '/login';
}

function goToDashboard() {
    // 로그인 체크
    fetch('/api/session')
        .then(response => response.json())
        .then(data => {
            if (data.logged_in) {
                // 이미 메인 페이지에 있으면 위험 예측 실행
                predictRisk();
            } else {
                alert('로그인이 필요한 서비스입니다.');
                goToLogin();
            }
        })
        .catch(() => {
            goToLogin();
        });
}

function showDemo() {
    alert('데모 기능은 준비 중입니다. 로그인 후 전체 서비스를 이용해보세요!');
    goToLogin();
}

function requireLogin(service) {
    alert(`${service} 서비스는 로그인 후 이용 가능합니다.`);
    goToLogin();
}

function showRegister() {
    alert('회원가입 기능은 준비 중입니다. 데모 계정으로 로그인해보세요!\n\nID: admin\nPW: 1234');
}

// ======================
// 애니메이션 및 효과
// ======================

function animateStats() {
    const stats = [
        { id: 'data-count', target: 15420, suffix: '' },
        { id: 'accuracy', target: 95.2, suffix: '%' }
    ];

    stats.forEach(stat => {
        const element = document.getElementById(stat.id);
        if (!element) return;

        let current = 0;
        const increment = stat.target / 100;
        const timer = setInterval(() => {
            current += increment;
            if (current >= stat.target) {
                current = stat.target;
                clearInterval(timer);
            }
            
            if (stat.suffix === '%') {
                element.textContent = current.toFixed(1) + stat.suffix;
            } else {
                element.textContent = Math.round(current).toLocaleString() + stat.suffix;
            }
        }, 20);
    });
}

// ======================
// 부드러운 스크롤
// ======================

function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ======================
// 네비게이션 활성화
// ======================

function initNavigationHighlight() {
    window.addEventListener('scroll', function() {
        const sections = ['home', 'services', 'about', 'community'];
        const navLinks = document.querySelectorAll('.nav-link');
        
        let current = '';
        sections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                const sectionTop = section.offsetTop;
                if (scrollY >= sectionTop - 200) {
                    current = sectionId;
                }
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ======================
// 실시간 업데이트
// ======================

function startRealTimeUpdates() {
    // 30초마다 상태 확인
    statusUpdateInterval = setInterval(checkStatus, 30000);
}

function stopRealTimeUpdates() {
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
}

// ======================
// 페이지 초기화
// ======================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🌊 CREW_SOOM 대시보드 초기화 시작...');
    
    // 부드러운 스크롤 초기화
    initSmoothScroll();
    
    // 네비게이션 하이라이트 초기화
    initNavigationHighlight();
    
    // 초기 상태 확인
    checkStatus();
    
    // 통계 애니메이션 시작 (1초 후)
    setTimeout(animateStats, 1000);
    
    // 실시간 업데이트 시작
    startRealTimeUpdates();
    
    // 초기 예측 실행 (예측 폼이 있는 경우)
    const precipInput = document.getElementById('precipitation');
    if (precipInput) {
        setTimeout(() => {
            predictRisk();
        }, 2000);
    }
    
    console.log('✅ 대시보드 초기화 완료!');
});

// 페이지 언로드 시 정리
window.addEventListener('beforeunload', function() {
    stopRealTimeUpdates();
});

// ======================
// 알림 스타일 CSS 추가
// ======================

const notificationStyles = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
        min-width: 300px;
        max-width: 500px;
    }
    
    .notification-content {
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .notification-icon {
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    .notification-message {
        flex: 1;
        white-space: pre-line;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .notification-close {
        background: none;
        border: none;
        font-size: 1.2rem;
        cursor: pointer;
        color: #666;
        flex-shrink: 0;
    }
    
    .notification-success {
        border-left: 4px solid #28a745;
    }
    
    .notification-error {
        border-left: 4px solid #dc3545;
    }
    
    .notification-warning {
        border-left: 4px solid #ffc107;
    }
    
    .notification-info {
        border-left: 4px solid #17a2b8;
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .status-connected {
        color: #28a745 !important;
        font-weight: 600;
    }
    
    .status-disconnected {
        color: #dc3545 !important;
        font-weight: 600;
    }
    
    .status-ready {
        color: #28a745 !important;
        font-weight: 600;
    }
    
    .status-not-ready {
        color: #ffc107 !important;
        font-weight: 600;
    }
    
    .fresh {
        color: #28a745 !important;
        font-weight: 600;
    }
    
    .stale {
        color: #dc3545 !important;
        font-weight: 600;
    }
    
    .viz-result {
        width: 100%;
    }
    
    .viz-image {
        width: 100%;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 16px;
    }
    
    .viz-info {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    
    .viz-info p {
        margin-bottom: 8px;
    }
    
    .viz-info p:last-child {
        margin-bottom: 0;
    }
`;

// 스타일 추가
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);