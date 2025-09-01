import pandas as pd


dasan1 = "datasets/민원(콜센터) 질의응답_다산콜센터_대중교통 안내_Training.json"
df1 = pd.read_json(dasan1)
df2 = pd.read_json(dasan1)

print(df1.columns)
print(df2.columns)

# 전체 데이터셋중 개체명에 "도시철도, 지하철" 이 있는 경우만 추출함.
