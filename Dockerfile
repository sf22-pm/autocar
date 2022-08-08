FROM ubuntu:20.04 as base
RUN apt-get update
RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install wget vim -y
WORKDIR /sf22
COPY . ./
RUN ./docker_setup.sh
