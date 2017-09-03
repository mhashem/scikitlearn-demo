FROM ubuntu:14.04

RUN apt-get update &&\
apt-get install -y pkg-config libopenblas-dev liblapack-dev git-core build-essential gfortran python-dev curl libfreetype6-dev libjpeg-dev libhdf5-dev liblzo2-dev libbz2-dev && \
python && \
pip install numpy && \
pip install scikit-image && \
pip install scipy && \
pip install scikit-learn && \
pip install Flask

# Bundle app source
ADD . /code

WORKDIR /code

EXPOSE  5000

CMD python rest.py