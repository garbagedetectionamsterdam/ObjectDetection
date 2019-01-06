#!/bin/bash
sudo nvidia-docker run -v /mnt/nfs:/mnt/nfs --name retrain_server -d garbagedetectionamsterdam/retrain_server
