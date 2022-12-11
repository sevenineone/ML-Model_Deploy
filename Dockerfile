FROM python:3.7

WORKDIR /app

COPY requirements.txt requirements.txt 

RUN pip install -r requirements.txt

COPY model model

COPY model.py model.py

COPY server.py server.py

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "server:app" ]