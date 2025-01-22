FROM bioconductor/bioconductor_docker
MAINTAINER Gabriel Reder gk@reder.io

# Install R packages with specific versions
RUN R -e "BiocManager::install(c('xcms', 'qvalue'))" && \
    R -e "install.packages('tidyverse', version='2.0.0')"

# Set environment variables
ENV PATH=/root/miniconda3/bin:/usr/local/bin/R:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib/R/lib:$LD_LIBRARY_PATH \
    PKG_CONFIG_PATH=/usr/local/lib/R/lib/pkgconfig/:$PKG_CONFIG_PATH

# Install system dependencies and Miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh && \
    /bin/bash /root/miniconda.sh -b -p /root/miniconda3 && \
    rm /root/miniconda.sh && \
    /root/miniconda3/bin/conda clean -tipy && \
    ln -s /root/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install Python packages with specific versions
RUN conda install -y python=3.12 && \
    conda install -y scikit-learn=1.5.2 pandas=2.2.3 && \
    conda install -y -c conda-forge ipdb=0.13.13 && \
    conda clean -afy

RUN pip install rpy2==3.5.11 \
                matplotlib==3.9.2 \
                lxml==5.3.0 \
                xlrd==2.0.1 \
                pyyaml==6.0.2 \
                tqdm==4.66.6 && \
    pip cache purge

# Copy the package source and install it
COPY . /app
WORKDIR /app
RUN pip install -e .

# Set final working directory
WORKDIR /home/scripts

# Add entrypoints to PATH
ENV PATH="/app/src/metabozen:${PATH}"