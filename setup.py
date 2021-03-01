import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

setuptools.setup(
      name='neural_analysis', 
      version='0.0.1', 
      description='Miller Lab Neural Analysis Resources', 
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/sbrincat/neural_analysis.git', 
      author= ['Scott Brincat', 'John Tauber'],
      author_email= ['sbrincat@mit.edu', 'jtauber@mit.edu'],
      license='LICENSE', 
      packages=setuptools.find_packages(),
      classifiers = [
          'Intended Audience :: Me',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS',
      ],
      zip_safe=False)