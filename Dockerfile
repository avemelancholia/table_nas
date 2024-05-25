FROM ubuntu:20.04
#FROMub  nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install wget vim git -y && apt-get update &&  \ 
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        /opt/conda/bin/conda clean -afy

ENV PATH /opt/conda/bin:$PATH
WORKDIR /home/
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
        pip install notebook seaborn  scikit-learn

RUN git clone https://github.com/automl/NASLib.git naslib && \
        cd naslib && \
        pip install --upgrade pip setuptools wheel && \
        pip install -e . && \
        mkdir /home/experiments/


CMD ["jupyter", "notebook", "--ip=0.0.0.0",  "--port=8899", "--allow-root", "--no-browser", "--NotebookApp.token=''",  "--NotebookApp.password=''"]


