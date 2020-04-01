FROM tiangolo/uwsgi-nginx-flask:python3.7

RUN pip install kaggle-environments numpy scipy gym

ARG AGENT_FILE
ENV AGENT_FILE=${AGENT_FILE}

COPY ./app /app
COPY ${AGENT_FILE} /app/agent.py