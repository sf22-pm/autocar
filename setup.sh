#!/bin/bash
sudo apt-get update
sudo apt-get install wget python-pip python3-pip -y
sudo apt install --no-install-recommends software-properties-common dirmngr -y
sudo wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
sudo apt update
sudo apt install --no-install-recommends r-base -y
sudo pip install -r requirements.txt
sudo R --no-save <<SHAR_EOF
install.packages('arulesCBA')
q()
SHAR_OEF
