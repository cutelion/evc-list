import os
import streamlit as st
import ast  # for converting embeddings saved as strings back to arrays
import pandas as pd  # for DataFrames to store article sections and embedding

# load excel files from current directory
# df = pd.read_csv("kr_evcharger_list.csv", header=2, usecols=range(15))

# # 모든 열에 rstrip() 적용
# evc = df.applymap(lambda x: x.rstrip() if isinstance(x, str) else x)

df = pd.read_csv("k-apt_info_20230818.csv", header=1, na_values=[''], keep_default_na=False)
filtered_df = df[df.apply(lambda row: pd.isnull(row['도로명주소']) 
#                          or not isinstance(row['도로명주소'], str) 
                          , axis=1)]
print("NaN 값 또는 문자열이 아닌 값을 가진 행:\n", filtered_df['도로명주소'], type(filtered_df['도로명주소']))
# 모든 열에 rstrip() 적용
# aptInfo = df.applymap(lambda x: x.rstrip() if isinstance(x, str) else x)

# count_addr = selected[selected['주소'].apply(lambda x: pd.isnull(x) or not isinstance(x, str))].shape[0]
# st.write(count_addr, "개의 주소가 없습니다.")