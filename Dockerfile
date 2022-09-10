FROM nvidia/cuda:11.7.0-devel-ubuntu22.04
WORKDIR /usr/src/app
RUN apt-get update
RUN apt-get install -y python3 python3-pip git
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
CMD sh pl_translate/train.sh
# CMD sh pl_translate/pred.sh
