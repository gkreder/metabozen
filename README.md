# PyMetab

## 201222 - For Caroline and Payam

I've been using this repo to share code with my experimental collaborator (Jakub, who is on here as well). I should note that this code has been written for Jakub to be able to run the pipeline, changing inputs and parameters without having to touch much command line or code. With that in mind, the way it's currently run is by editing the input .xlsx files then running the python commands below. It's not quite camera-ready, but hopefully should be readable. 

The xcms.py script is just a Python wrapper around XCMS. The bulk of the code is contained in clustering.py which takes in XCMS feature output and
runs our parent-ion identification clustering pipeline on it. 

All of the dependencies for running both XCMS and the clustering pipeline are contained in the docker image [gkreder/py_metab](https://hub.docker.com/r/gkreder/py_metab). In fact, this same code itself is also located in that docker image in the directory /home/pymetab so the whole thing could just be run by pulling the docker image and running a container - no need to clone this repo if you don't want to. 

If you would like to try running the pipeline and seeing the output, you could try running (from inside /home/pymetab): 

```python clustering.py --in_file clustering.xlsx```

Right now, the pipeline has a high memory overhead. The run on this example data should require about 8GB of RAM and about 10 minutes. Making overhead lower is on the list of improvements but hasn't been a priority since memory is not an issue on the servers we use. 

Very briefly, the pipeline clusters XCMS features into groups ideally coming from a single parent ion or a couple of closely related ones. The clustering involves first computing a hierarchical clustering on the features using a distance metric we wrote. For two features f_i and f_j we have a distance of 

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=d(f_i,f_j) = (1- R_{i,j}) %2B \alpha \left(1 - \exp\left(\frac{-\rho_{i,j}}{\tau}\right)\right)">    
<img src="https://render.githubusercontent.com/render/math?math=\rho_{i,j} = \sqrt{\frac{1}{n} \sum_{k = 1}^{n} (t_{i,k} - t_{j,k})^2}">    
<img src="https://render.githubusercontent.com/render/math?math=R_{i,j} = \text{Pearson Correlation between feature i and feature j calculated across overlapping samples}">    
<img src="https://render.githubusercontent.com/render/math?math=t_{i,k} = \text{Retention time of feature i for sample k}">  

</p>

Tree cuts are computed on this dendrogram by traversing down the tree and cutting at points where the resulting sub-trees satisfy the condition that at least *frac_peaks* (default = 0.8) of the children are within *rt_1sWindow* (default = 5 seconds) of the sub-tree root. After this, some filtering is done by removing cluster outliers and reassigning cluster parents when there exists some cluster child with a higher m/z and at least *parent_mz_check_intensity_frac* (default = 0.6) average intensity compared to the current parent. 

The clustering process is then repeated iteratively on the parents only. Any two parents that cluster together have their respective clusters merged into one. This process is repeated until the number of clusters is stable. 

Please let me know if you have any feedback on this or would like to chat more about it! Happy to answer any questions or to talk ideas. You can reach me at gkreder@gmail.com

-Gabe

## 201101 - For Jakub

To launch Docker, from your docker Powershell:
```docker run -it -v <DATA_DIRECTORY>:/home/data/ <SCRIPTS_DIRECTORY>:/home/scripts gkreder/py_metab bash```

To run XCMS from inside Docker /home/scripts:
```python xcms.py --in_file xcms.xlsx```

To run clustering from inside Docker /home/scripts:
```python clustering.py --in_file clustering.xlsx```

