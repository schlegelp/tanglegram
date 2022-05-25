tanglegram
==========
Uses scipy and matplotlib to plot simple tanglegrams. Inspired by the amazing [dendextend](https://github.com/talgalili/dendextend) by Tal Galili.

## Installation
First, get [PIP](https://pip.pypa.io/en/stable/installing/) and then run in terminal:

```
pip3 install tanglegram -U
```

To install the bleeding-edge version from Github you can run:

```
pip3 install git+https://github.com/schlegelp/tanglegram@master
```

#### Dependencies
Installing via [PIP](https://pip.pypa.io/en/stable/installing/) should install all external dependencies. You may run into problems on Windows though. In that case, you need to install dependencies manually, here is a list of dependencies (check out `install_requires` in [setup.py](https://raw.githubusercontent.com/schlegelp/PyMaid/master/setup.py) for version info):

- [Pandas](http://pandas.pydata.org/)
- [SciPy](http://www.scipy.org)
- [Numpy](http://www.scipy.org)
- [Matplotlib](http://www.matplotlib.org)
- [tqdm](https://github.com/tqdm/tqdm)

## How it works

`tanglegram` exposes three functions:

1. `tanglegram.plot` plots a tanglegram (optionally untangling)
2. `tanglegram.entanglement` measures the entanglement between two linkages
3. `tanglegram.untangle` rotates dendrograms to minimize entanglement

```Python
import tanglegram as tg
import matplotlib.pyplot as plt
import pandas as pd

# Generate two distance matrices and just switch labels in one
labelsA= ['A', 'B', 'C', 'D']
labelsB= ['B', 'A', 'C', 'D']
data = [[ 0,  .1,  .4, .3],
        [.1,   0,  .5, .6],
        [.4,  .5,   0, .2],
        [.3,  .6,  .2,  0]]

mat1 = pd.DataFrame(data,
                    columns=labelsA,
                    index=labelsA)

mat2 = pd.DataFrame(data,
                    columns=labelsB,
                    index=labelsB)

# Plot tanglegram
fig = tg.plot(mat1, mat2, sort=False)
plt.show()
```

<img src="https://user-images.githubusercontent.com/7161148/105351954-2ae19f80-5be5-11eb-9dad-2dd0fe83d44d.png" width="650">

```Python
# Plot again but this time try minimizing cross-over
fig = tg.plot(mat1, mat2, sort=True)
plt.show()
```

<img src="https://user-images.githubusercontent.com/7161148/105351772-e8b85e00-5be4-11eb-9343-db42f143ec68.png" width="650">


## Known Issues:
* layout does not scale well, i.e. small dendrograms look weird

## License:
This code is under GNU GPL V3
