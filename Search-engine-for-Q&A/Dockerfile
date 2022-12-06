FROM python:3.6-slim-buster

COPY ./src /app

WORKDIR /app

RUN pip3.6 install --upgrade pip

RUN pip3.6 install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3.6","searchES_FlaskAPI.py"]