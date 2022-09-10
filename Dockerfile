FROM nvidia/cuda:11.7.0-devel-ubuntu22.04
WORKDIR /usr/src/app
RUN apt-get update
RUN apt-get install -y python3 python3-pip git
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
# CMD python3 ja2en_translate/src/train.py --data_mode JESC
# CMD python3 ja2en_translate/src/pred.py --data_mode JESC
CMD python3 Seq_VAE/src/train.py --data_file data/NTT/persona.txt