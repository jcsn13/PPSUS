FROM python:3.7.14-bullseye
ARG port

USER root
COPY . /server
WORKDIR /server

ENV PORT=$port

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
    && apt-get -y install curl \
    && apt-get install libgomp1

RUN chgrp -R 0 /server \
    && chmod -R g=u /server \
    && pip install pip --upgrade \
    && pip install -r requirements.txt

CMD uvicorn app:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 180