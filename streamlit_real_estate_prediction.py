#起動方法　streamlit run streamlit_real_estate_prediction.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from requests.exceptions import JSONDecodeError
import locale


def custom_format_currency(num_str):
    f=float(num_str)
    amount=int(f)
    trillion = amount // 1000000000000
    billion = (amount // 100000000) % 10000
    million = (amount // 10000) % 10000
    yen = amount % 10000

    formatted_amount = ""
    if trillion > 0:
        formatted_amount += f"{trillion}兆"
    if billion > 0:
        formatted_amount += f"{billion}億"
    if million > 0:
        formatted_amount += f"{million}万"
    if yen > 0:
        formatted_amount += f"{yen}円"
    return formatted_amount


# 仮のデータフレームを作成
data = {
    "Nearest_Station_Name": ["飯田橋", "東京", "新宿", "渋谷", "秋葉原", "有楽町"],
    "City_Ward_Town_Name": ["千代田区", "千代田区", "新宿区", "渋谷区", "千代田区", "千代田区"],
    "District_Name": ["飯田橋", "大手町", "西新宿", "神南", "秋葉原", "有楽町"]
}
df = pd.DataFrame(data)

# Streamlitアプリのタイトルと説明
st.title("不動産取引価格予測できたらアプリ")
st.write("以下のフォームに情報を入力して、算出ボタンを押してください。")

# 最寄駅名の入力と検索
station_query = st.text_input("最寄駅名を検索")
filtered_stations = df[df['Nearest_Station_Name'].str.contains(station_query)]

# 最寄駅名から選択
selected_station = st.selectbox("最寄駅名を選択", filtered_stations['Nearest_Station_Name'].unique())

# 現在の年を取得
current_year = datetime.now().year
# リストを生成
years_list = [str(year) + "年" for year in range(current_year, 1944, -1)]    


# 市区町村名と地区名をフィルタリング
if selected_station:
    possible_cities = df[df['Nearest_Station_Name'] == selected_station]
    city_ward_town_name = st.selectbox("市区町村名", possible_cities['City_Ward_Town_Name'].unique())
    district_name = st.selectbox("地区名", possible_cities[possible_cities['City_Ward_Town_Name'] == city_ward_town_name]['District_Name'].unique())

# 入力フォーム
with st.form("prediction_form"):
    nearest_station_distance_min = st.number_input("最寄駅：距離（分）", min_value=0, value=5)
    floor_area_m2 = st.number_input("延床面積（㎡）", min_value=10, value=25)
    year_built = st.selectbox("建築年", options=years_list)
    building_structure = st.selectbox("建物の構造", options=["ＲＣ","木造","ＳＲＣ","鉄骨造","その他","軽量鉄骨造"])
    intended_use = st.selectbox("用途", options=["住宅", "共同住宅"])
    front_road_type = st.selectbox("前面道路：種類",options=["国道", "公道", "私道","その他"])            
    front_road_width_m = st.number_input("前面道路：幅員（ｍ）", min_value=2, value=4)
    #transaction_date = st.text_input("取引時期")
    submit_button = st.form_submit_button("算出")

# APIエンドポイントのURL
url = "http://127.0.0.1:8000/predict/"

# フォームが送信されたときの処理
if submit_button:
    request_data = {
        "City_Ward_Town_Name": city_ward_town_name,
        "District_Name": district_name,
        "Nearest_Station_Name": selected_station,
        "Nearest_Station_Distance_min": nearest_station_distance_min,
        "Transaction_Price_Total": 0,
        "Floor_Area_m2": floor_area_m2,
        "Year_Built": year_built,
        "Building_Structure": building_structure,
        "Intended_Use": intended_use,
        "Front_Road_Type": front_road_type,
        "Front_Road_Width_m": front_road_width_m,
        "Transaction_Date": "2023年第4四半期"
    }
    response = requests.post(url, json=request_data)
    if response.status_code == 200:
        try:
            prediction = response.json().get("prediction")
            formatted_number = custom_format_currency(prediction)
            st.success(f"予測された取引価格: {formatted_number}")
        except JSONDecodeError:
            st.error("予測の取得に失敗しました。JSON形式のエラーです。")
            st.write(f"Raw Response: {response.text}")  # Raw レスポンスを表示
    else:
        st.error(f"APIエラー: {response.status_code}")
        st.text(f"詳細: {response.text}")  # エラーの詳細を表示