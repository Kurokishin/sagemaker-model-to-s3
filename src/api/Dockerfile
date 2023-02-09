FROM python:3.10

ADD . /app

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

ENV FLASK_APP=main.py

CMD ["flask", "run", "--host=0.0.0.0"]