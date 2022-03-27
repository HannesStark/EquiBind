# FROM ubuntu:focal
FROM --platform=linux/amd64 nvidia/cuda:11.6.0-devel-ubuntu20.04 as base

# installing packages required for installation
RUN echo "downloading basic packages for installation"
RUN apt-get update
RUN apt-get install -y tmux wget curl git
RUN apt-get install -y libstdc++6 gcc

# checking installation of GPU tools
RUN gcc --version
RUN nvcc --version

# install conda
RUN wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /conda
RUN rm Miniconda3-latest-Linux-x86_64.sh
RUN . "/conda/etc/profile.d/conda.sh"
ENV PATH="/conda/condabin:${PATH}"
# RUN conda create --name colabfold-conda python=3.7 -y
RUN mkdir lab-equibind
COPY environment_cpuonly.yml /lab-equibind/.
RUN conda env create -f lab-equibind/environment_cpuonly.yml
# Switch to the new environment:
SHELL ["conda", "run", "-n", "equibind", "/bin/bash", "-c"] 
RUN conda update -n base conda -y
COPY . /lab-equibind/.
# RUN source /conda/etc/profile.d/conda.sh
WORKDIR /lab-equibind
# CMD ["conda", "activate", "equibind-conda"]

# run tests
FROM base as test
RUN python inference.py --config=configs_clean/inference.yml
