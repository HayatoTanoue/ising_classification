# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ENV DEBCONF_NOWARNINGS yes
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    wget since apt-utils

RUN apt-get install -y git libgl1-mesa-dev
RUN conda install -c conda-forge ipywidgets nodejs
RUN conda install -c conda-forge jupyterlab=2.1 ujson=1.35 jedi=0.15.2 parso=0.5.2 python-language-server r-languageserver

RUN pip install --upgrade pip
RUN pip install matplotlib networkx pandas seaborn sklearn
RUN pip install opencv-python opencv-contrib-python

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]
