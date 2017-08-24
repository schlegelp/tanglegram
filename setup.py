from distutils.core import setup

import re


VERSIONFILE="tanglegram/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setup(
    name='tanglegram',
    version=verstr,
    packages=['tanglegram',],
    license='GNU GPL V3',
    description='Plots simple tanglegrams from two dendrograms',
    long_description=open('README.md').read(),
    url = 'https://github.com/schlegelp/tanglegram',
    author='Philipp Schlegel',
    author_email = 'pms70@cam.ac.uk',
    keywords='python tanglegram dendrogram',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],    

    install_requires=[        
        "scipy>=0.18.1",        
        "numpy>=1.12.1",
        "matplotlib>=2.0.0",        
    ],

    python_requires='>=3',

    zip_safe = False
)
