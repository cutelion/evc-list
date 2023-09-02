import os
import streamlit as st
import pandas as pd  # for DataFrames to store article sections and embedding
from fuzzywuzzy import fuzz
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_api_key = os.environ.get("OPENAI_API_KEY")

# 아파트 이름과 주소 유사도 비교
def split_address(addr):
    addrs = addr.split(",")  # 쉼표를 기준으로 주소를 나눈다.
    addrs = [addr.rsplit("(", 1)[0].strip() for addr in addrs]  # 괄호 안의 내용 제거
    bases_nums = [addr.rsplit(" ", 1) if " " in addr else (addr, "") for addr in addrs]
    return bases_nums

def get_address_similarity(addr1, addr2):
    bases_nums1 = split_address(addr1)
    bases_nums2 = split_address(addr2)
    
    max_similarity = 0
    
    for base1, num1 in bases_nums1:
        for base2, num2 in bases_nums2:
            num_similarity = fuzz.ratio(num1, num2)     # 번지수를 문자열로 비교
            # 앞부분은 토큰 비교 (OO동 있어도 됨), partial_ratio 쓰면 빈 칸 무시하기에는 좋으나 동이 들어있으면 낮게 나옴
            base_similarity = fuzz.token_set_ratio(base1, base2)
            addr_similarity = 0.4 * num_similarity + 0.6 * base_similarity
            max_similarity = max(max_similarity, addr_similarity)
    
    return max_similarity


def get_name_similarity(name1, name2):
    return fuzz.token_set_ratio(name1, name2)

def get_name_addr_similarity(data1, data2):
    name1, addr1 = data1
    name2, addr2 = data2
    
    name_similarity = get_name_similarity(name1, name2)
    address_similarity = get_address_similarity(addr1, addr2)

    if address_similarity == 100 or (name_similarity == 100 and address_similarity > 70):
        total_similarity = 100
    else:
        total_similarity = 0.4 * name_similarity + 0.6 * address_similarity
    return total_similarity, name_similarity, address_similarity

# 이름과 주소 유사도 비교 using OpenAI's Sentence Embedding
def get_oai_similarity(data1, data2):
    name1, addr1 = data1
    name2, addr2 = data2
    
    text1 = f"이름: {name1}, 주소: {addr1}"
    text2 = f"이름: {name2}, 주소: {addr2}"

    db = FAISS.from_texts([text1], OpenAIEmbeddings())
    similarity = db.similarity_search_with_relevance_scores(text2)
    return similarity[0][1] # similarity score

# 주소가 유사한 아파트 찾기
@st.cache_data
def get_matches(selEvc, selApt, THRESHOLD=75):
    matches = []
    # Loop through each 주소 in selEvc    
    for i, row in selEvc.iterrows():
        best_match = {'단지코드': "NA", '단지명': "NA", '비교주소': "NA", '매칭점수': 0, '이름비교': 0, '주소비교': 0}
        # Loop through each 도로명주소 in selApt
        for j, row2 in selApt.iterrows():
            score, name_score, address_score = get_name_addr_similarity((row['충전소'], row['주소']), (row2['단지명'], row2['도로명주소']))

            # If the score is greater than 90 and better than the previous best match, update the best match
            if score > best_match['매칭점수']:
                best_match = {
                    '단지코드': row2['단지코드'],
                    '단지명': row2['단지명'],
                    '비교주소': row2['도로명주소'],
                    '매칭점수': score,
                    '이름비교': name_score,
                    '주소비교': address_score
                }
                if score < THRESHOLD:
                    best_match['단지코드'] = "NA"

        # Append the best match 단지코드 to the list
        matches.append(best_match)
        
    # 새로운 칼럼들을 selEvc DataFrame에 추가
    for key in matches[0].keys():
        selEvc[key] = [match[key] for match in matches]

    compare_result = {
        '주소일치': len(selEvc[selEvc['매칭점수'] == 100]),
        '유사추정': len(selEvc[(selEvc['매칭점수'] < 100) & (selEvc['단지코드'] != "NA")]),
        '불일치': len(selEvc[selEvc['단지코드'] == "NA"])
        }
    return selEvc, compare_result


@st.cache_data
def load_datafile():
    # load excel files from current directory
    
    with st.spinner('첫 실행 시 사례 데이터를 로딩합니다. 잠시만 기다려주세요...'):
        df = pd.read_csv("kr_evcharger_list.csv", header=2, usecols=range(15))
        df['주소'] = df['주소'].fillna('Unknown')
        # 모든 열에 rstrip() 적용
        evc = df.applymap(lambda x: x.rstrip() if isinstance(x, str) else x)
        df = pd.read_csv("k-apt_info_20230818.csv", header=1)
        df['도로명주소'] = df['도로명주소'].fillna('Unknown')
        # 모든 열에 rstrip() 적용
        df = df[['시도', '시군구', '단지코드', '단지명', '도로명주소', '총주차대수', '세대수']]
        aptInfo = df.applymap(lambda x: x.rstrip() if isinstance(x, str) else x)
        
        st.success('Data Loaded!')
    return evc, aptInfo

@st.cache_data
def load_data_parquet():
    with st.spinner('첫 실행 시 사례 데이터를 로딩합니다. 잠시만 기다려주세요...'):
        evc = pd.read_parquet("evc-list.parquet")
        aptInfo = pd.read_parquet("apt-list.parquet")
        # st.success('Data Loaded!')
    return evc, aptInfo


st.title("충전 데이터 분석")
"""
https://www.ev.or.kr/evmonitor 에서 받은 충전소 데이터에서 시설구분(소)가 '아파트'인 데이터를 가져옵니다.
충전소 이름으로 묶어서 충전기 수를 계산하고, K-APT에서 받은 단지 기본 정보를 이용하여 아파트 주소와 유사한 충전소를 찾습니다.
"""

# load parquet of (충전소 현황, 단지 기본 정보) from current directory
evc, aptInfo = load_data_parquet()

# Set up sidebar
THRESHOLD = st.sidebar.slider("(단지명 및 주소) 유사도 임계값", 60, 100, 75, 5)

# Study on the number of charging stations in each Apt
st.sidebar.header("아파트 충전기 설치 현황")

filtered_df = evc[evc['시설구분(소)'] == '아파트']

# If there are any rows satisfying the condition
if not filtered_df.empty:
    # Group by '충전소' and count the entries
    grouped = filtered_df.groupby('충전소').count()
    st.sidebar.write("'충전소'가 있는 아파트 총수: ", len(grouped))
    st.sidebar.write("k-apt 등록 아파트 총수: ", len(aptInfo))
    # st.write(grouped)
    
# 아파트별 충전기 데이터 새로 만들기
aptEvc = filtered_df.groupby('충전소').agg({
    '운영기관': 'first',
    '충전기ID': 'size',  # 충전기수 계산
    '충전기타입': 'first',
    '시설구분(대)': 'first',
    '시설구분(소)': 'first',
    '지역': 'first',
    '시군구': 'first',
    '주소': 'first',
}).reset_index()
# 열 이름 변경
aptEvc.rename(columns={'충전기ID': '충전기수'}, inplace=True)


# save files to parquet
# if st.button("Save Data"):
#     evc.to_parquet("evc-list.parquet")
#     aptInfo.to_parquet("apt-list.parquet")



# study on # of Apt in each region 
# 지역 선택한 것만 보이도록 selectbox 만들기

option1 = ['전체 선택'] + list(aptEvc['지역'].unique())
selected_province = st.sidebar.selectbox('시도를 선택하세요', option1, index=min(1, len(option1)-1))
if selected_province == '전체 선택':
    option2 = ['전체 선택']
    selected_province = None
else:
    option2 = ['전체 선택'] + list(aptEvc[aptEvc['지역'] == selected_province]['시군구'].unique())

selected_region = st.sidebar.selectbox('시군구를 선택하세요', option2, index=min(1, len(option2)-1))
if selected_region == '전체 선택':
    selected_region = None

st.sidebar.warning("전체선택을 하면 처리하는 데 시간이 꽤 걸립니다. 주의하세요.")

# 선택된 시군구 관련 데이터 표시하기 위한 sidebar
sidebar_selected = st.sidebar.container()

view_raw = st.expander("시도/시군구별 데이터 현황 (충전소 & 단지 기본 정보)", expanded=False)

with view_raw:
    # 두 칼럼으로 충전소 데이터와 아파트 데이터 보여주기
    colEvc, colApt = st.columns(2)

    with colEvc:
        if selected_province and selected_region:
            selected = aptEvc[(aptEvc['지역'] == selected_province) & (aptEvc['시군구'] == selected_region)]
        elif selected_province:
            selected = aptEvc[aptEvc['지역'] == selected_province]
        else:
            selected = aptEvc
            
        selEvc = selected[['충전소', '주소', '충전기수', '지역', '시군구']]
        
        st.write("충전소 현황", len(selected))
        st.write(selected)
        with sidebar_selected:
            st.write("충전소수(ev.or.kr)", len(selected))
        # count_addr = selected[selected['주소'].apply(lambda x: pd.isnull(x) or not isinstance(x, str))].shape[0]
        # st.write(count_addr, "개의 주소가 없습니다.")

    with colApt:
        if selected_province and selected_region:
            selected = aptInfo[(aptInfo['시도'] == selected_province) & (aptInfo['시군구'] == selected_region)]
        elif selected_province:
            selected = aptInfo[aptInfo['시도'] == selected_province]
        else:
            selected = aptInfo
        
        selApt = selected[['단지명', '단지코드', '도로명주소', '총주차대수', '세대수', '시도', '시군구']]
        # selected = selApt.sort_values(by='단지명')

        st.write("아파트 현황", len(selected))
        st.write(selected)
        with sidebar_selected:
            st.write("아파트수(k-apt)", len(selected))

        # count_addr = selected[selected['도로명주소'].apply(lambda x: pd.isnull(x))].shape[0]
        # st.write(count_addr, "개의 주소가 없습니다.")





view_context = st.expander("아파트 이름 & 주소 유사도 검색 결과", expanded=False)

# selEvc, compare_result = get_matches(selEvc)
# 각 그룹의 결과를 저장할 빈 리스트를 생성합니다.
result_list = []
compare_result = {'주소일치': 0, '유사추정': 0, '불일치': 0}

# 각 그룹에 대해 get_matches 함수를 반복적으로 호출합니다.
for (region, district), group in selEvc.groupby(['지역', '시군구']):
    for (r2, d2), g2 in selApt.groupby(['시도', '시군구']):
        if (region, district) == (r2, d2):
            selEvc_group, compare_result_group = get_matches(group, g2, THRESHOLD)
            result_list.append(selEvc_group)
            st.info(f"{region} {district}의 데이터를 처리했습니다.")
            for key in compare_result.keys():
                compare_result[key] += compare_result_group[key]

# 각 그룹의 결과를 연결하여 최종 결과 데이터프레임을 만듭니다.
if result_list:
    result_df = pd.concat(result_list, ignore_index=True)
    selEvc = result_df

with sidebar_selected:
    st.write("주소 일치 하는 경우가", compare_result['주소일치'], "개,\n비슷하게 추정한 경우가", compare_result['유사추정'], "개,\n불일치하는 경우가", compare_result['불일치'], "개 입니다.")

with view_context:
    st.write("주소 일치", compare_result['주소일치'], selEvc[selEvc['매칭점수'] == 100])
    st.write("이름 및 주소 유사", compare_result['유사추정'], selEvc[(selEvc['매칭점수'] < 100) & (selEvc['단지코드'] != "NA")])
    st.write("이름 및 주소 불일치", compare_result['불일치'], selEvc[selEvc['단지코드'] == "NA"])



# 단지코드별로 groupby해서 전체 주차장수와 충전기수, 비율을 구하기
df = selEvc[selEvc['단지코드'] != "NA"]
df = df.groupby('단지코드').agg({'충전기수': 'sum'}).reset_index()
df = df.merge(selApt, on='단지코드', how='left')
df['충전기설치율'] = df['충전기수'] / df['총주차대수']

st.markdown("""## 단지별 충전기 설치 현황
K-APT의 단지 정보와 일치(또는 유사)한 경우 단지코드별로 합산하여 계산했습니다.""")

st.dataframe(
    df[['단지코드', '단지명', '총주차대수', '충전기수', '충전기설치율']].sort_values(by='충전기설치율', ascending=False),
    column_config={
        "충전기설치율": st.column_config.ProgressColumn(
            "충전기설치율",
            min_value=0,
            max_value=0.2,
            format="%.2f",
        ),
    },
    hide_index=True,
)

df_unmatched = selEvc[selEvc['단지코드'] == "NA"]
st.markdown("""## 단지코드가 없는 충전소
K-APT의 단지 정보와 유사하지 않아 개별적으로 표시하였습니다.""")
st.dataframe(
    df_unmatched[['충전소', '주소', '충전기수']],
    hide_index=True,
)

# st.header("충전소 주소와 아파트 주소가 완전히 일치하지 않는 경우")
# df = selEvc[(selEvc['매칭점수'] < 100) & (selEvc['매칭점수'] > 0)]
# st.table(df[['매칭점수', '이름비교', '충전소', '단지명', '주소비교', '주소', '비교주소']].sort_values(by='매칭점수', ascending=False))
# st.write("충전소 주소와 아파트 주소가 완전히 일치하지 않는 경우: ", len(df))


# df = selEvc[(selEvc['매칭점수'] < 90) & (selEvc['주소비교'] > 50)].head(5)

# OpenAI Embedding을 이용한 유사도 비교
# for i, row in df.iterrows():
#     oai_similarity = get_oai_similarity((row['충전소'], row['주소']), (row['단지명'], row['비교주소']))
#     st.write(row, oai_similarity)


#for row in df.iterrows():

# '매칭점수'가 100점 미만인 경우에 한해서만 실행




# st.table(df[['매칭점수', '이름비교', '충전소', '단지명', '주소비교', '주소', '비교주소']].sort_values(by='매칭점수', ascending=False))

# st.table(selEvc[(selEvc['매칭점수'] < 100) & (selEvc['매칭점수'] > 0)])

#            st.write(row['충전소'], row['주소'], row2['단지명'], row2['도로명주소'], fuzz.ratio(row['주소'], row2['도로명주소']))





