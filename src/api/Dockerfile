FROM python:3.10

ADD . /app

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "main.py"]