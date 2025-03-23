from setuptools import setup, find_packages

import re


VERSIONFILE = "tanglegram/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

setup(
    name='tanglegram',
    version=verstr,
    packages=find_packages(),
    license='GNU GPL V3',
    description='Plot simple tanglegrams from two dendrograms',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/schlegelp/tanglegram',
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='python tanglegram dendrogram',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',

    ],

    install_requires=requirements,

    python_requires='>=3.9',

    zip_safe=False
)
