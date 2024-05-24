FROM bioconductor/bioconductor_docker
MAINTAINER Gabriel Reder gk@reder.io


RUN R -e "BiocManager::install('xcms')"
RUN R -e "install.packages('tidyverse')"
RUN R -e "BiocManager::install('qvalue')"

ENV PATH /root/miniconda3/bin:$PATH
ENV PATH /usr/local/bin/R:$PATH
ENV LD_LIBRARY_PATH /usr/local/lib/R/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH /usr/local/lib/R/lib/pkgconfig/:$PKG_CONFIG_PATH


RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh
RUN /bin/bash /root/miniconda.sh -b -p /root/miniconda3
RUN rm /root/miniconda.sh
RUN /root/miniconda3/bin/conda clean -tipy && \
    ln -s /root/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc


RUN conda install -y scikit-learn
RUN conda install -y pandas
RUN conda install -y -c conda-forge ipdb
# RUN conda install jupyterlab
# RUN pip install ipywidgets
# RUN jupyter nbextension enable --py widgetsnbextension
# RUN conda install -c conda-forge nodejs
# RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN pip install rpy2
RUN pip install matplotlib
RUN pip install lxml pyteomics
RUN pip install xlrd pyyaml
