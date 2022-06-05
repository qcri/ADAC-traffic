#!/bin/bash
# Install sumo and set env variables

mkdir 3rdparty
cd 3rdparty
wget -O sumo.zip https://prdownloads.sourceforge.net/sumo/sumo-src-1.2.0.zip?download
unzip sumo-src-1.2.0.zip
cd sumo-1.2.0/
make && make install
