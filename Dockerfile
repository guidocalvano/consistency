FROM nvcr.io/nvidia/tensorflow:19.07-py3

COPY ./requirements_base.txt /app/requirements_base.txt
COPY ./requirements_docker.txt /app/requirements_docker.txt

WORKDIR /app
RUN echo `pwd`
RUN pip install -r ./requirements_docker.txt

COPY input /app/input
COPY learning /app/learning
COPY *.py /app/
COPY *.json /app/

