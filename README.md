tanglegram
==========
Uses scipy and matplotlib to plot simple tanglegrams. Inspired by the amazing [dendextend](https://github.com/talgalili/dendextend) by Tal Galili.

## Installation
I recommend using [Python Packaging Index (PIP)](https://pypi.python.org/pypi) to install the package.
First, get [PIP](https://pip.pypa.io/en/stable/installing/) and then run in terminal:  

`pip install git+git://github.com/schlegelp/tanglegram@master`  

This command should also work to update the package.

**Attention**: on Windows, the dependencies (i.e. Numpy, Pandas and SciPy) will likely fail to install automatically. Your best bet is to get a Python distribution that already includes them (e.g. [Anaconda](https://www.continuum.io/downloads)). 

If your default distribution is Python 2, you have to explicitly tell [PIP](https://pip.pypa.io/en/stable/installing/) to install for Python 3:

`pip3 install git+git://github.com/schlegelp/tanglegram@master`  

#### External libraries used:
Installing via [PIP](https://pip.pypa.io/en/stable/installing/) should install all external dependencies. You may run into problems on Windows though. In that case, you need to install dependencies manually, here is a list of dependencies (check out `install_requires` in [setup.py](https://raw.githubusercontent.com/schlegelp/PyMaid/master/setup.py) for version info):

- [Pandas](http://pandas.pydata.org/)
- [SciPy](http://www.scipy.org)
- [Numpy](http://www.scipy.org) 
- [Matplotlib](http://www.matplotlib.org)
- [Tqdm](https://pypi.python.org/pypi/tqdm)

## Quickstart:

```python
from tanglegram import tangle
import numpy as np
import matplotlib.pyplot as plt

# Quick test: 
# Generate two distance matrices and just switch labels in one
labelsA= ['A','B','C','D']
labelsB= ['B','A','C','D']
data = np.array( [ [ 1, .1,  0, 0],
                 [.1,  1, .5, 0],
                 [ 0, .5,  1, 0],
                 [ 0,  0,  0, 1]
                ])

mat1 = pd.DataFrame( data,
                     columns=labelsA,
                     index=labelsA)

mat2 = pd.DataFrame( data,
                     columns=labelsB,
                     index=labelsB)

# Plot tanglegram
fig = tangle.plot(mat1,mat2)
plt.show()
```

<img src="https://user-images.githubusercontent.com/7161148/29683302-c2cc22e0-8905-11e7-9091-97e55bce1ddb.png" width="650">

## Known Issues:
* layout does not scale well, i.e. small dendrograms look weird

## License:
This code is under GNU GPL V3
