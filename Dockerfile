FROM ubuntu:20.04

ARG USER_ID
ARG GROUP_ID

# Needed for a silent cmake install
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Add the host's user/group ID so that the host can easily access files created by the guest
# Checks if group_id already exists (thanks Romeo)
RUN if ! grep -q ${GROUP_ID} /etc/group ; then groupadd -g ${GROUP_ID} dev_user; fi
RUN useradd -l -u ${USER_ID} -g ${GROUP_ID} onair_dev
# Add a home directory for the user
RUN mkdir /home/onair_dev && \
    chown onair_dev /home/onair_dev

# TODO: add ssh with X forwarding

# Install
RUN \
  apt-get update && \
  apt-get -y upgrade

# Bare minimum to build/run cFS
RUN \
  apt-get install sudo && \
  apt-get install -y build-essential && \
  apt-get install -y gcc-multilib && \
  apt-get install -y git && \
  apt-get install -y cmake && \
  apt-get install -y xterm

# lcov: needed for cFS unit tests
# xxd: does hex dumps, just plain handy to have
RUN \
  apt-get install -y lcov && \
  apt-get install -y xxd

# OnAIR Dependencies
RUN \
  apt-get install -y wget

# Ensure that all packages are up to date after new packages have been added above
RUN \
  apt-get update && \
  apt-get -y upgrade && \
  rm -rf /var/lib/apt/lists/*

# Add user to sudoers so that they can up the mqueue depth
RUN adduser onair_dev sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER onair_dev

# Install miniconda
ENV CONDA_DIR /home/onair_dev/conda
RUN \
  mkdir -p $CONDA_DIR && \
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
  bash ~/miniconda.sh -b -u -p $CONDA_DIR && \
  rm -rf ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Make OnAir requirements file accessible by onair_dev user
COPY environment.yml /home/onair_dev/environment.yml
RUN \
  . $CONDA_DIR/etc/profile.d/conda.sh && \
  conda init bash && \
  . ~/.bashrc && \
  conda env create -f /home/onair_dev/environment.yml && \
  conda activate onair

# Make sure that the onair conda environment is loaded
RUN \
  echo "conda activate onair" >> ~/.bashrc
