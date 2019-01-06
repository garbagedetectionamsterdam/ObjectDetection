#!/bin/bash
sudo nvidia-docker run -p 5000:5000  -v /mnt/nfs:/mnt/nfs --name prediction_api -d garbagedetectionamsterdam/prediction_api
