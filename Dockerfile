FROM python:3.9-slim

# Ref:
# * https://medium.com/dsights/streamlit-deployment-on-google-cloud-serverless-container-platform-1a8330d29062

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=

COPY . /app

CMD streamlit run evc-list.py --server.port=${PORT}  --browser.serverAddress="0.0.0.0"