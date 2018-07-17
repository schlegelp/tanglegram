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
import tanglegram as tg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate two distance matrices and just switch labels in one
labelsA= ['A', 'B', 'C', 'D']
labelsB= ['B', 'A', 'C', 'D']
data = [[ 0,  .1,  .4, .3],
        [.1,   0,  .5, .6],
        [ .4, .5,   0, .2],
        [ .3, .6,  .2,  0]]

mat1 = pd.DataFrame(data,
                    columns=labelsA,
                    index=labelsA)

mat2 = pd.DataFrame(data,
                    columns=labelsB,
                    index=labelsB)

# Plot tanglegram
fig = tg.gen_tangle(mat1, mat2, optimize_order=False)
plt.show()

# Plot again but this time try minimizing cross-over
fig = tg.gen_tangle(mat1, mat2, optimize_order=1000)
plt.show()
```

<img src="https://user-images.githubusercontent.com/7161148/42809649-fde86b00-89ad-11e8-9ecd-051f40731bc1.png" width="650">

## Known Issues:
* layout does not scale well, i.e. small dendrograms look weird

## License:
This code is under GNU GPL V3
