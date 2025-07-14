import os
import requests

def download_seoul_geojson():
    url = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo.json"
    save_path = "static/data/seoul_gu.geojson"

    os.makedirs("static/data", exist_ok=True)

    if not os.path.exists(save_path):
        print("🌐 서울시 GeoJSON 다운로드 중...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print("✅ GeoJSON 다운로드 완료:", save_path)
        else:
            print("❌ 다운로드 실패:", response.status_code)
    else:
        print("✅ GeoJSON 이미 존재:", save_path)

if __name__ == "__main__":
    download_seoul_geojson()
