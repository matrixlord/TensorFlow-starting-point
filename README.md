# TensorFlow starting point

This repo contains a starting point for a new TensorFlow application.
It is build around Docker. When you first run it, it will download and
compile TensorFlow for your CPU. If your CPU supports SSE4.2, AVX2 etc,
it will run faster from the ready made TensorFlow bin.

It also contains a sample neural network to test the your new TensorFlow
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

Go to /scripts folder and run
~~~~
python tf.py
~~~~

# By SocialNerds
* [SocialNerds.gr](https://www.socialnerds.gr/)
* [YouTube](https://www.youtube.com/SocialNerdsGR)
* [Facebook](https://www.facebook.com/SocialNerdsGR)
* [Twitter](https://twitter.com/socialnerdsgr)