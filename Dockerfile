FROM debian:stable-slim

#This is referencing a directory inside the container
WORKDIR /home/
RUN apt-get update -y
RUN apt-get install -y	python3 \
						python3-pip \
						vim \
						git \
						tmux

RUN pip install qiskit
RUN pip install mapomatic
RUN pip install black
RUN pip install matplotlib
CMD ["bash"]

# 1. Build the container: `docker build -t qos .`
# 2. Start the container: `docker run -v $(pwd):/home:z -it qos`