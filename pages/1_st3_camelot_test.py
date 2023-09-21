import os
import openai
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains import create_extraction_chain
from dotenv import load_dotenv
import streamlit as st
import camelot
from datetime import datetime, date, time
import pandas as pd
import json


# env_file = ktcb.env_path
# _ = load_dotenv(ktcb.env_path)
# openai.api_key  = os.getenv('OPENAI_API_KEY')


# get file path: changed from loading from ktcb_directory.py
base_path = 'pages'
data_direc = os.path.join(base_path, 'data')
pickle_raw = os.path.join(base_path, 'pickle_raw')

pdf_files = ['기술신용 조사서_1번 PoC_1번 샘플.pdf','기술신용 조사서_1번 PoC_2번 샘플.pdf']

# get api key from streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_api_key = os.environ.get("OPENAI_API_KEY")

# GPT 모델 선택
gpt3 = ChatOpenAI(model_name = "gpt-3.5-turbo-0613", temperature=0.0)
gpt3_16 = ChatOpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0.0)
gpt4 = ChatOpenAI(model_name = "gpt-4", temperature=0.0)
llm = gpt3

model_names = ["gpt3", "gpt3_16", "gpt4"]

selected_model_name = st.sidebar.selectbox("Select a LLM model", model_names, index=0)

if selected_model_name == "gpt3_16":
    llm = gpt3_16
elif selected_model_name == "gpt3":
    llm = gpt3
else:
    llm = gpt4


# define schema and prompt for extraction chain: default는 schema만 있으면 된다.
schema = {
    "properties": {
        "성명":{"type":"string"},
        "생년월일":{"type":"string", "description":"birth date"},
        "최종학력":{"type":"string","description":"total number of years"},
        "동업계 경력":{"type":"number"},
        "대표자유형":{"type":"string","enum":["대표자", "경영실권자"]},
        "경영형태":{"type":"string", "enum":["창업", "2세승계"]},
        "전공":{"type":"string"},
        "자격증":{"type":"string"},
        "주요 경력":{"description":"list of main experiences"}
    }
}

# you may remove the examples and prompt if you want to use the default schema
examples = [
    {
        "성명": "김정호",
        "생년월일": "1993.03.15",
        "최종학력": "대졸",
        "동업계 경력":"3",
        "대표자유형":"경영실권자",
        "경영형태":"2세승계",
        "전공":"화학공학과",
        "자격증":"",
        "주요 경력":{'근무기간': '2020.11.~2022.03.', '근무처': 'Fun Company', '업종': '제조', '담당업무': '경영지원', '직위': '임직원'}
    }
]


sys_prompt_text = """
Your goal is to extract the necessary data from the data frame that matches the schema described below.
When extracting information, make sure it matches the properties information exactly.
Do not make up any data. If the data unavailable, leave the section blank.

Return the result in a json format and do not add any attributes that do not appear as shown in the schema below.
{schema}

Example extractions are below. Follow the format given in the example exactly.
{examples}
"""

sys_prompt = PromptTemplate(
    template= sys_prompt_text,
    input_variables=["schema", "examples"]
)
human_prompt = PromptTemplate(
    template= "{text}",
    input_variables=["text"]
)
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=sys_prompt),
    HumanMessagePromptTemplate(prompt=human_prompt)]
)


# run extraction chain
@st.cache_data
def run_extraction_chain(doc) -> dict:     # doc이 list라서 hash하지 않도록 _doc으로 바꿈
    default_chain = create_extraction_chain(schema, llm)
    new_chain = create_extraction_chain(schema, llm)
    new_chain.prompt = chat_prompt

    result = {
        "default": default_chain.run(doc),
        "new": new_chain.run({"schema": schema, "examples": examples, "text": doc})
    }

    return result

# change extracted json to dataframe
def json_to_df(json_data) -> dict:
    df = {
        "기본정보": pd.DataFrame(json_data).transpose().drop("주요 경력"),
        "경력": pd.DataFrame(json_data[0]["주요 경력"])
    }
    return df

# compare two dataframes in streamlit: 눈으로 보기 좋게 만들었다.
def compare_result(doc_result):
    result_default = json_to_df(doc_result["default"])
    result_new = json_to_df(doc_result["new"])
    
    st.write("기본 정보 비교")
    main_compare = pd.concat([result_default["기본정보"], result_new["기본정보"]], axis=1)
    main_compare.columns = ["DEFAULT", "NEW: prompted"]
    st.write(main_compare)
    
    st.write("경력 비교, N:__는 new chain에서 추출한 정보")
    career_new = result_new["경력"].rename(columns=lambda x: f"N:{x}")
    career_compare = pd.concat([result_default["경력"], career_new], axis=1)
    st.write(career_compare)


# get pdf files from data directory
loader = PyPDFDirectoryLoader(data_direc + "/")
docs = loader.load()


# st.title("KTCB Table Data Extraction")
# """
# extract chain을 비교합니다.
# 1. default chain은 schema만 있으면 되고, new chain은 examples를 참고하고 system prompt에 따로 넣도록 조정한 것입니다.
# 하지만 결과는 큰 차이가 나지 않습니다.
# 2. pdf를 text로 파싱한 것과 camelot으로 table을 추출한 것을 비교합니다. sampe1에서는 깔끔하게 추출되지 않음.
# """

# st.header("Data extraction from parsed pdf text")
# for doc in docs:
#     st.write(doc.metadata['source'])
#     doc_result = run_extraction_chain(doc.page_content)
#     st.json(doc_result, expanded=False)
#     compare_result(doc_result)

st.header("Data extraction from tables by Camelot")
"""
camelot 이용의 경우 1번 샘플에서 여러 json으로 나눠지는 문제가 있다.
그래서 table을 합쳐 비교하는 compare_result 함수를 사용하지 않았다.
"""
for pdf_file in pdf_files:
    pdf_file_path = os.path.join(data_direc, pdf_file) 
    tables = camelot.read_pdf(pdf_file_path, line_scale=30)
    for table in tables:
        st.write(pdf_file_path)
        st.write(table.df)
        table_result = run_extraction_chain(table.df)
        st.json(table_result, expanded=False)
        # compare_result(table_result)


# st.sidebar.header("Usage for today")
# date_today = date.today()
# with st.sidebar.expander("Usage for today"):
#     st.write(openai.api_requestor.APIRequestor().request("GET", "/usage?date="+str(date_today))[0].data)