#起動方法　ターミナルで　uvicorn fastAPI_real_estate_prediction:app --reload
import pandas as pd
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse


app = FastAPI()
from real_estate_prediction_model import prediction_service,use_columns
#['市区町村名','地区名','最寄駅：名称','最寄駅：距離（分）','取引価格（総額）','延床面積（㎡）','建築年','建物の構造','用途','前面道路：種類','前面道路：幅員（ｍ）','取引時期']

import logging
# ログの設定
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# ロガーの取得
logger = logging.getLogger("uvicorn.error")

mapping_dict = {
    'City_Ward_Town_Name': '市区町村名',
    'District_Name': '地区名',
    'Nearest_Station_Name': '最寄駅：名称',
    'Nearest_Station_Distance_min': '最寄駅：距離（分）',
    'Floor_Area_m2': '延床面積（㎡）',
    'Year_Built': '建築年',
    'Building_Structure': '建物の構造',
    'Intended_Use': '用途',
    'Front_Road_Type': '前面道路：種類',
    'Front_Road_Width_m': '前面道路：幅員（ｍ）',
    'Transaction_Price_Total':'取引価格（総額）',
    'Transaction_Date': '取引時期'
}

class InputData(BaseModel):
    City_Ward_Town_Name: str #'市区町村名' ex:千代田区
    District_Name: str #'地区名' ex:飯田橋
    Nearest_Station_Name: str#'最寄駅：名称' ex:飯田橋
    Nearest_Station_Distance_min:int #'最寄駅：距離（分）' ex:4    
    Floor_Area_m2:int #'延床面積（㎡）'　ex:220
    Year_Built:str #'建築年' ex:2007年
    Building_Structure:str #'建物の構造' ex:ＲＣ
    Intended_Use:str #'用途' ex:住宅 or　共同住宅
    Front_Road_Type:str #'前面道路：種類'　ex:区道
    Front_Road_Width_m:int#'前面道路：幅員（ｍ）' ex:4
    Transaction_Price_Total:int=0 #'取引価格（総額）' （省略すること）
    Transaction_Date:str ='2023年第4四半期' #'取引時期' ex:2023年第2四半期
    
    
# トップページ
@app.get("/")
async def index():
    return {"message": "不動産価格を推論するAPIです。"}

@app.post("/predict/")
async def make_predictions(features: InputData):  
    try:
        data = features.dict() 
        df = pd.DataFrame([data])
        df.rename(columns=mapping_dict, inplace=True)        
        prediction = prediction_service(df)[0]
        return {"prediction": str(prediction)}
    except Exception as e:
        # エラーをログに記録
        logger.error("An error occurred during prediction: %s", str(e), exc_info=True)        
        return JSONResponse(status_code=500, content={"message": "Internal server error"})

def test_main():
    #テスト用
    Input_Data=InputData()
    Input_Data.City_Ward_Town_Name='千代田区'
    Input_Data.District_Name='飯田橋'
    Input_Data.Nearest_Station_Name='飯田橋'
    Input_Data.Nearest_Station_Distance_min=4
    Input_Data.Transaction_Price_Total=0
    Input_Data.Floor_Area_m2=220
    Input_Data.Year_Built='2007年'
    Input_Data.Building_Structure='ＲＣ'
    Input_Data.Intended_Use='共同住宅'
    Input_Data.Front_Road_Type='区道'
    Input_Data.Front_Road_Width_m=4
    Input_Data.Transaction_Date='2023年第4四半期'
    
    test_data={        
        '市区町村名':[Input_Data.City_Ward_Town_Name],
        '地区名':[Input_Data.District_Name],
        '最寄駅：名称':[Input_Data.Nearest_Station_Name],
        '最寄駅：距離（分）':[Input_Data.Nearest_Station_Distance_min],
        '取引価格（総額）':[Input_Data.Transaction_Price_Total],
        '延床面積（㎡）':[Input_Data.Floor_Area_m2],
        '建築年':[Input_Data.Year_Built],
        '建物の構造':[Input_Data.Building_Structure],
        '用途':[Input_Data.Intended_Use],
        '前面道路：種類':[Input_Data.Front_Road_Type],
        '前面道路：幅員（ｍ）':[Input_Data.Front_Road_Width_m],
        '取引時期':[Input_Data.Transaction_Date]
        }
    df = pd.DataFrame(test_data)
    print({'prediction':str(prediction_service (df)[0])})
    return
if __name__ == '__main__':
    test_main()

