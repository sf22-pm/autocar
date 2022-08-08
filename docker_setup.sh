#!/bin/bash
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends software-properties-common dirmngr -y
DEBIAN_FRONTEND=noninteractive wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
DEBIAN_FRONTEND=noninteractive add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
DEBIAN_FRONTEND=noninteractive apt update
DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends r-base -y
DEBIAN_FRONTEND=noninteractive apt install python3-pip lib64gfortran-10-dev libopenblas64-dev liblapack-pic -y
DEBIAN_FRONTEND=noninteractive pip install -r requirements.txt
R --no-save <<SHAR_EOF
install.packages('arulesCBA')
q()
SHAR_OEF
