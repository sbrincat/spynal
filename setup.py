import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

requirements = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'h5py',
    'hdf5storage',
    'patsy',
    'scikit-learn',
    'torch'
    ]

setuptools.setup(
    name='spynal',
    version='0.1.3',
    description='Simple Python Neural Analysis Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sbrincat/spynal.git',
    author= ['Scott Brincat', 'John Tauber'],
    author_email= 'sbrincat@mit.edu',
    license='LICENSE',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
    ],
    zip_safe=False)
