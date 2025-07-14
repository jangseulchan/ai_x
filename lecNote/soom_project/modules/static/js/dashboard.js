// modules/static/js/dashboard.js

let statusUpdateInterval;

// 전역 로딩 함수
function showGlobalLoading(message = '처리 중...') {
    document.getElementById('loading-message').textContent = message;
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideGlobalLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// 상태 확인 및 업데이트
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        // 오늘 날짜 표시
        if (status.today) {
            document.getElementById('today-date').textContent = `📅 오늘: ${status.today}`;
            document.getElementById('prediction-date').value = status.today;
        }
        
        // 시스템 상태
        const statusDiv = document.getElementById('system-status');
        let statusText = `📊 데이터: ${status.data_loaded ? '✅ 로드됨' : '❌ 없음'} | `;
        statusText += `🤖 모델: ${status.model_loaded ? '✅ 로드됨' : '❌ 없음'} | `;
        statusText += `🌤️ API: ${status.api_available ? '✅ 연결됨' : '❌ 키 없음'}`;
        if (status.api_location) {
            statusText += ` (${status.api_location})`;
        }
        statusDiv.innerHTML = statusText;
        statusDiv.className = (status.data_loaded && status.model_loaded && status.api_available) ? 'status status-success' : 'status status-warning';
        
        // 데이터 정보 카드 업데이트
        document.getElementById('data-rows').textContent = status.data_rows || '-';
        
        if (status.data_start_date && status.data_end_date) {
            const startDate = new Date(status.data_start_date).toLocaleDateString();
            const endDate = new Date(status.data_end_date).toLocaleDateString();
            document.getElementById('data-period').textContent = `${startDate} ~ ${endDate}`;
        } else {
            document.getElementById('data-period').textContent = '-';
        }
        
        if (status.data_last_updated) {
            const lastUpdate = new Date(status.data_last_updated);
            const now = new Date();
            const diffMinutes = Math.floor((now - lastUpdate) / 60000);
            
            if (diffMinutes < 5) {
                document.getElementById('last-update').innerHTML = `<span class="fresh">${diffMinutes}분 전</span>`;
            } else if (diffMinutes < 60) {
                document.getElementById('last-update').textContent = `${diffMinutes}분 전`;
            } else {
                const diffHours = Math.floor(diffMinutes / 60);
                if (diffHours < 24) {
                    document.getElementById('last-update').innerHTML = `<span class="stale">${diffHours}시간 전</span>`;
                } else {
                    document.getElementById('last-update').innerHTML = `<span class="stale">${lastUpdate.toLocaleDateString()}</span>`;
                }
            }
        } else {
            document.getElementById('last-update').textContent = '-';
        }
        
        document.getElementById('model-status').textContent = status.model_loaded ? '활성화' : '미훈련';
        
        // 자동 업데이트 상태
        const autoUpdateToggle = document.getElementById('auto-update-toggle');
        const updateIndicator = document.getElementById('update-indicator');
        const autoUpdateStatus = document.getElementById('auto-update-status');
        
        autoUpdateToggle.checked = status.auto_update_enabled;
        if (status.auto_update_enabled) {
            updateIndicator.className = 'update-indicator update-active';
            autoUpdateStatus.textContent = '활성화';
        } else {
            updateIndicator.className = 'update-indicator update-inactive';
            autoUpdateStatus.textContent = '비활성화';
        }
        
        // 마지막 체크 시간
        const lastCheckSpan = document.getElementById('last-check');
        if (status.last_check_time && status.auto_update_enabled) {
            const lastCheck = new Date(status.last_check_time);
            const checkDiffSeconds = Math.floor((now - lastCheck) / 1000);
            lastCheckSpan.textContent = `(마지막 체크: ${checkDiffSeconds}초 전)`;
        } else {
            lastCheckSpan.textContent = '';
        }
        
    } catch (error) {
        document.getElementById('system-status').innerHTML = '❌ 시스템 오류';
        document.getElementById('system-status').className = 'status status-error';
    }
}

// 자동 업데이트 토글
async function toggleAutoUpdate() {
    try {
        const response = await fetch('/api/toggle_auto_update', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            checkStatus(); // 상태 즉시 업데이트
        } else {
            alert(`❌ ${result.message}`);
        }
    } catch (error) {
        alert('자동 업데이트 설정 오류: ' + error.message);
    }
}

// 데이터 로드
async function loadData() {
    showGlobalLoading('데이터를 로드하고 있습니다...');
    try {
        const response = await fetch('/api/load_data', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            alert(`✅ ${result.message}\\n기간: ${result.start_date} ~ ${result.end_date}`);
            checkStatus();
        } else {
            alert(`❌ ${result.message}`);
        }
    } catch (error) {
        alert('데이터 로드 오류: ' + error.message);
    }
    hideGlobalLoading();
}

// 데이터 업데이트
async function updateData() {
    showGlobalLoading('실제 API 데이터를 가져오고 있습니다...');
    try {
        const response = await fetch('/api/update_data', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            let message = `✅ ${result.message}`;
            if (result.data_source) {
                message += `\\n데이터 소스: ${result.data_source}`;
            }
            if (result.precipitation !== undefined) {
                message += `\\n현재 강수량: ${result.precipitation}mm`;
                message += `\\n온도: ${result.temperature}°C`;
                message += `\\n습도: ${result.humidity}%`;
            }
            message += `\\n이전: ${result.old_count}행 → 현재: ${result.new_count}행`;
            alert(message);
            checkStatus();
        } else {
            alert(`❌ ${result.message}`);
        }
    } catch (error) {
        alert('데이터 업데이트 오류: ' + error.message);
    }
    hideGlobalLoading();
}

// 모델 훈련
async function trainModel() {
    showGlobalLoading('모델을 훈련하고 있습니다...');
    try {
        const response = await fetch('/api/train_model', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            alert(`✅ ${result.message}\\nAUC: ${result.auc}, 정밀도: ${result.precision}\\n훈련 데이터: ${result.training_data_size}행`);
            checkStatus();
        } else {
            alert(`❌ ${result.message}`);
        }
    } catch (error) {
        alert('모델 훈련 오류: ' + error.message);
    }
    hideGlobalLoading();
}

// 시각화 생성
async function createVisualization(type) {
    showGlobalLoading(`${type} 차트를 생성하고 있습니다...`);
    try {
        const response = await fetch('/api/create_visualization', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: type })
        });
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('visualization-area').innerHTML = 
                `<img src="${result.image}" class="viz-image" alt="${type} 차트">
                 <p class="realtime-info">데이터 수: ${result.data_count}개</p>`;
        } else {
            alert(`❌ ${result.message}`);
        }
    } catch (error) {
        alert('시각화 오류: ' + error.message);
    }
    hideGlobalLoading();
}

// 침수 위험 예측
async function predictRisk() {
    const data = {
        precipitation: parseFloat(document.getElementById('precipitation').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        avg_temp: parseFloat(document.getElementById('temperature').value),
        precip_sum_3d: parseFloat(document.getElementById('precip_3d').value),
        season_type: document.getElementById('season').value,
        target_date: document.getElementById('prediction-date').value  // 선택된 날짜
    };
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        
        document.getElementById('risk-display').className = `risk-meter risk-${result.risk_level}`;
        document.getElementById('risk-display').innerHTML = `
            ${result.risk_color} ${result.risk_name}<br>
            <div style="font-size: 36px; margin: 10px 0;">${result.risk_score}점</div>
            ${result.action}
        `;
        
        document.getElementById('recommendations').innerHTML = `
            <h4>📋 권장 행동:</h4>
            <ul>${result.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
        `;
        
        // 예측 메타 정보
        const predictionTime = new Date(result.prediction_time).toLocaleString();
        let freshnessInfo = '';
        if (result.data_freshness !== null) {
            const freshnessMinutes = Math.floor(result.data_freshness);
            if (freshnessMinutes < 10) {
                freshnessInfo = `<span class="fresh">데이터 신선도: 매우 좋음 (${freshnessMinutes}분 전)</span>`;
            } else if (freshnessMinutes < 60) {
                freshnessInfo = `데이터 신선도: 좋음 (${freshnessMinutes}분 전)`;
            } else {
                freshnessInfo = `<span class="stale">데이터 신선도: 주의 (${Math.floor(freshnessMinutes/60)}시간 전)</span>`;
            }
        }
        
        document.getElementById('prediction-meta').innerHTML = `
            <p><strong>예측 날짜: ${result.prediction_date}</strong></p>
            <p>예측 시간: ${predictionTime}</p>
            <p>사용 모델: ${result.model_used}</p>
            <p>${freshnessInfo}</p>
        `;
        
    } catch (error) {
        alert('예측 오류: ' + error.message);
    }
}

// 테스트 시나리오
const scenarios = {
    'calm': {precipitation: 0, humidity: 60, avg_temp: 20, precip_sum_3d: 0, season_type: 'dry'},
    'light': {precipitation: 15, humidity: 75, avg_temp: 22, precip_sum_3d: 25, season_type: 'rainy'},
    'medium': {precipitation: 35, humidity: 85, avg_temp: 24, precip_sum_3d: 60, season_type: 'rainy'},
    'heavy': {precipitation: 80, humidity: 95, avg_temp: 26, precip_sum_3d: 120, season_type: 'rainy'},
    'extreme': {precipitation: 130, humidity: 96, avg_temp: 26, precip_sum_3d: 200, season_type: 'rainy'}
};

function testScenario(scenarioName) {
    const scenario = scenarios[scenarioName];
    document.getElementById('precipitation').value = scenario.precipitation;
    document.getElementById('humidity').value = scenario.humidity;
    document.getElementById('temperature').value = scenario.avg_temp;
    document.getElementById('precip_3d').value = scenario.precip_sum_3d;
    document.getElementById('season').value = scenario.season_type;
    predictRisk();
}

// 페이지 로드 시 초기화
window.onload = function() {
    checkStatus();
    predictRisk();
    
    // 5초마다 상태 업데이트
    statusUpdateInterval = setInterval(checkStatus, 5000);
};

// 페이지 언로드 시 정리
window.onbeforeunload = function() {
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
};