FROM python:3.12-slim

WORKDIR /prediction_server

COPY requirements.txt requirements.txt
RUN pip3 install -r ./requirements.txt

COPY . /prediction_server 

EXPOSE 5000

ENV FLASK_APP="__init__:create_app"

CMD [ "python3", "-u", "-m" , "flask", "run", "--host=0.0.0.0"]
