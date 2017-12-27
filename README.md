# TensorFlow starting point

This repo contains a starting point for a new TensorFlow application.
It is built around Docker. When you first run it, it will download and
compile TensorFlow for your CPU. If your CPU supports SSE4.2, AVX2 etc,
it will run faster compared to the ready made TensorFlow bin.

It also contains a sample neural network to test your new TensorFlow
container.

# Installation
Requirements
- You need to have [Docker](https://docs.docker.com/engine/installation/) installed

# Run

Run in root folder,
~~~~
docker-compose build && docker-compose up -d
~~~~

Login to the container,
~~~~
docker exec -it ai /bin/bash -c "TERM=$TERM exec bash"
~~~~

Go to /data folder and run
~~~~
python tf.py
~~~~

# Some things to consider

This is used as a starting point for machine learning projects. For this reason,
Keras and some other libraries come pre-installed. You may remove them at your
own discretion.

The model is trained with a handful of photos, that I personally shot to avoid copyright infringement stuff. You may improve the accuracy by adding more photos.

# Maintainer
[Thanos Nokas](https://www.linkedin.com/in/thanosnokas)