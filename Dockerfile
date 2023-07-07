ARG BASE_IMAGE_TYPE=cpu
# these images have been pushed to Dockerhub but you can find
# each Dockerfile used in the `base_images` directory 
FROM jafermarq/jetsonfederated_$BASE_IMAGE_TYPE:latest

RUN apt-get install wget -y

# Download and extract CIFAR-10
# To keep things simple, we keep this as part of the docker image.
# If the dataset is already in your system you can mount it instead.
#ENV DATA_DIR=/app/data/cifar-10
ENV DATA_DIR=/app/data/ecg_data
RUN mkdir -p $DATA_DIR
WORKDIR $DATA_DIR
COPY ecg_data $DATA_DIR
#RUN wget https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz 
#RUN tar -zxvf cifar-10-python.tar.gz

WORKDIR /app
# Scripts needed for Flower client
ADD client.py /app
ADD utils.py /app


RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3.7 -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --config python3

# update pip
RUN python3 -m pip install --upgrade pip

# making sure the latest version of flower is installed
RUN pip3 install flwr>=1.0.0
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install Pillow
RUN pip3 install h5py

ENTRYPOINT ["python3","-u","./client.py"]
