FROM python:3.11

# Ref:
# * https://medium.com/dsights/streamlit-deployment-on-google-cloud-serverless-container-platform-1a8330d29062

RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=

COPY . /app

CMD streamlit run app.py --server.port=${PORT}  --browser.serverAddress="0.0.0.0"