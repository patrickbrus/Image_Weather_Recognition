# Image_Weather_Recognition
This repository contains code for the Image Weather Recognition App. The goal of this app is to create a webpage where the user can enter an image containing 
information about the weather and a trained keras model makes a prediction using the provided image. The classified class is then sent back to the webpage. In addition, one 
should be able to see a summary on all already classified classes (using a Redis Server in the backend).

The following steps were used to create the application (all code is included in this repository):

1. A Deep Learning model was trained on classifying the image into its weather class. Different CNNs were tested and different image classification strategies were tried 
   (i.e. augmentation, ...)
2. A flask application was created containing hosting of the front-end (templates/predict.html) and the backend (application.py). The front-end was created using codepen, where
   the html code is directly translated into a webpage. This really speeds up the coding process. 
3. A docker container was created to host the flask application. Port 5000 is exposed.
4. Docker-Compose was used to create the full application (Flask application with Redis Server). 
5. Kubernetes is used to create a cluster for the complete application. Below you can find the design chart of the Kubernetes cluster and all Services that were used.

## Kubernetes Application Design Chart
![alt text](https://github.com/patrickbrus/Image_Weather_Recognition/blob/master/figures/kubernetes_design_flowchart.png)
