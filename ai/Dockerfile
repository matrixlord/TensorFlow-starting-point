FROM ubuntu:16.04

# Get user id and group id from arguments
ARG UID
ARG GID

# Create user.
RUN groupadd -r serveruser -g $GID && useradd -ms /bin/bash serveruser -u $UID -g $GID

# Install some packages.
RUN apt-get update && apt-get install -y --no-install-recommends git openjdk-8-jdk curl \
    python-numpy python-dev python-pip python-wheel python-setuptools patch && \
    pip install --upgrade pip


# Install bazel.
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends bazel && \
    apt-get upgrade -y bazel

# Install TensorFlow.
WORKDIR /
RUN git clone https://github.com/tensorflow/tensorflow && \
    cd tensorflow && ./configure

# Build pip package.
WORKDIR /tensorflow
RUN bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Install pip package.
RUN pip --no-cache-dir install --upgrade /tmp/tensorflow_pkg/tensorflow-*.whl && \
    rm -rf /tmp/tensorflow_pkg && \
    rm -rf /root/.cache

# Install Keras.
RUN pip install keras

# Install h5py
RUN pip install h5py

# Copy keys and create main path.
RUN mkdir /home/serveruser/ai && chown -R serveruser:serveruser /home/serveruser

WORKDIR /home/serveruser/ai

EXPOSE 80 443 8888 6006

# Just to keep it running.
CMD tail -f /dev/null
