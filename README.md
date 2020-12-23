# PyMetab

## 201222 - For Caroline and Payam

I have been using this repo to share code with my experimental collaborator (Jakub, who is on here as well). I should note that this code has been written for Jakub to be able to run the pipeline, changing inputs and parameters without having to touch much command line or code. With that in mind, the way it's currently run is by editing the input .xlsx files then running the python commands below. It's not quite camera-ready, but hopefully should be readable. 

The xcms.py script is just a Python wrapper around XCMS. The bulk of the code is contained in clustering.py which takes in XCMS feature output and
runs our parent-ion identification clustering pipeline on it. 

All of the dependencies for running both XCMS and the clustering pipeline are contained in the docker image [gkreder/py_metab](https://hub.docker.com/r/gkreder/py_metab). In fact, this same code itself is also located in that docker image in the directory /home/pymetab so the whole thing could just be run by pulling the docker image and running a container - no need to clone this repo if you don't want to. 

If you would like to try running the pipeline and seeing the output, you could try running (from inside /home/pymetab):   
```python clustering.py --in_file clustering.xlsx```

Right now, the pipeline has a high memory overhead. The run on this example data should require about 8GB of RAM and about 10 minutes. Making overhead lower is on the list of improvements but hasn't been a priority since memory is not an issue on the servers we use. 

Very briefly, the pipeline clusters XCMS features into groups ideally coming from a single parent ion or a couple of closely related ones. The clustering involves first computing a hierarchical clustering on the features using a distance metric we wrote. For two features f_i and f_j we have a distance of 

```math
d(f_i, f_j) = (1- R_{i, j} + \alpha (1 - \exp(-\rho_{i, j} / \tau)))

\rho_{i, j} = \sqrt{\frac{1}{n} \sum_{k = 1}^{n} (t_{i,k} - t_{j, k})^2}

```
<img src="https://render.githubusercontent.com/render/math?math=d(f_i, f_j) = (1- R_{i, j} + \alpha (1 - \exp(-\rho_{i, j} / \tau)))">    
<img src="https://render.githubusercontent.com/render/math?math=\rho_{i, j} = \sqrt{\frac{1}{n} \sum_{k = 1}^{n} (t_{i,k} - t_{j, k})^2}">    
<img src="https://render.githubusercontent.com/render/math?math=t_{i, k} = \text{Retention time of feature $i$ for sample $k$}">    



## 201101 - For Jakub

To launch Docker, from your docker Powershell:
```docker run -it -v <DATA_DIRECTORY>:/home/data/ <SCRIPTS_DIRECTORY>:/home/scripts gkreder/py_metab bash```

To run XCMS from inside Docker /home/scripts:
```python xcms.py --in_file xcms.xlsx```

To run clustering from inside Docker /home/scripts:
```python clustering.py --in_file clustering.xlsx```

