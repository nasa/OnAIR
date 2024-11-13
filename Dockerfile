FROM ubuntu:22.04

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
  apt-get install -y python3.11 && \
  apt-get install -y python3.11-dev && \
  apt-get install -y python3-pip

# Add new packages to install here to prevent re-running previous instructions

# Ensure that all packages are up to date after new packages have been added above
RUN \
  apt-get update && \
  apt-get -y upgrade && \
  rm -rf /var/lib/apt/lists/*

# Add user to sudoers so that they can up the mqueue depth (for cFS)
RUN adduser onair_dev sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Make OnAir requirements files accessible by onair_dev user
COPY requirements-dev.txt /home/onair_dev/requirements-dev.txt
COPY requirements-run.txt /home/onair_dev/requirements-run.txt
COPY requirements-unittest.txt /home/onair_dev/requirements-unittest.txt
COPY requirements-lint.txt /home/onair_dev/requirements-lint.txt
RUN chown onair_dev /home/onair_dev/requirements*

USER onair_dev

# Python stuff is being installed for the local user
ENV PATH="${PATH}:/home/onair_dev/.local/bin"

# Install OnAIR deps
RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN python3.11 -m pip install --user -r /home/onair_dev/requirements-dev.txt
