FROM python:3.13-slim

WORKDIR /code

RUN apt-get update && apt-get install -y procps && \
    pip install -U pip && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r ./requirements.txt

COPY ./main.py /code/

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]