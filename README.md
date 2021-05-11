# PyMetab

## 201101 - For Jakub

To launch Docker, from your docker Powershell:
```docker run -it -v <DATA_DIRECTORY>:/home/data/ <SCRIPTS_DIRECTORY>:/home/scripts gkreder/py_metab bash```

To run XCMS from inside Docker /home/scripts:
```python xcms.py --in_file xcms.xlsx```

To run clustering from inside Docker /home/scripts:
```python clustering.py --in_file clustering.xlsx```

