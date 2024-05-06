#起動方法　streamlit run streamlit_real_estate_prediction.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from requests.exceptions import JSONDecodeError
import locale
from real_estate_prediction_model import prediction_service,use_columns
import os

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
df=pd.read_pickle(os.getcwd() + '/st_df.pkl')


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


# フォームが送信されたときの処理
if submit_button:
    request_data = {
        '市区町村名': [city_ward_town_name],
        '地区名': [district_name],
        '最寄駅：名称': [selected_station],
        '最寄駅：距離（分）': [nearest_station_distance_min],
        '取引価格（総額）': [0],
        '延床面積（㎡）': [floor_area_m2],
        '建築年': [year_built],
        '建物の構造': [building_structure],
        '用途': [intended_use],
        '前面道路：種類': [front_road_type],
        '前面道路：幅員（ｍ）': [front_road_width_m],
        '取引時期': ["2023年第4四半期"]
    }
    df = pd.DataFrame(request_data)
    prediction = str(prediction_service(df)[0])
    formatted_number = custom_format_currency(prediction)
    st.success(f"予測された取引価格: {formatted_number}")
