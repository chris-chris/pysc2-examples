FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER chris.ai <chris.ai@navercorp.com>

USER root

WORKDIR /root

# Change default python 2.7 => 3
RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install wget
RUN apt-get update
RUN apt-get install wget unzip git python-mpi4py cmake libopenmpi-dev -y

# Install StarCraftII
RUN wget -q http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.0.2.zip

# Uncompress StarCraftII
RUN unzip -P iagreetotheeula SC2.4.0.2.zip

# Download StarCraftII Maps
RUN wget -q https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip 

RUN wget -q http://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip
RUN wget -q http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season3.zip
RUN wget -q http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season2.zip
RUN wget -q http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season1.zip

# Uncompress zip files
RUN unzip mini_games.zip -d ~/StarCraftII/Maps/
RUN unzip -P iagreetotheeula Melee.zip -d ~/StarCraftII/Maps/
RUN unzip -P iagreetotheeula Ladder2017Season3.zip -d ~/StarCraftII/Maps/
RUN unzip -P iagreetotheeula Ladder2017Season2.zip -d ~/StarCraftII/Maps/
RUN unzip -P iagreetotheeula Ladder2017Season1.zip -d ~/StarCraftII/Maps/

# Delete zip files
RUN rm SC2.4.0.2.zip
RUN rm mini_games.zip
RUN rm Melee.zip
RUN rm Ladder2017Season3.zip
RUN rm Ladder2017Season2.zip
RUN rm Ladder2017Season1.zip

# Make Directory
RUN mkdir -p /home/nsml/

# Change permissions
RUN chmod -R 777 /home/nsml

# Move StarCraftII to /home/nsml
RUN mv ~/StarCraftII /home/nsml/