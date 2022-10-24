Installation
============

Users can install the most recent stable release using pip or install the latest updates from the Github repository

pip install
-----------
This is recommended for end users who prefer a simple installation and don't require the latest updates

- Open a terminal window, navigate to folder you want repository to live under (cd "parent directory")
- Run: pip install spynal

Install from source (Github)
----------------------------
This is recommended for developers and for end users that would like access to the latest updates

Download repository
^^^^^^^^^^^^^^^^^^^

    - Open a terminal window, navigate to folder you want repository to live under (cd "parent directory")
    - Run: git clone https://github.com/sbrincat/spynal.git

Install package
^^^^^^^^^^^^^^^

    - Navigate into the newly-created spynal folder (cd "parent directory"/spynal)
    - For end users:  at command line, run: python setup.py install
    - For developers: at command line, run: python setup.py develop
        (this will allow you to edit the library without reinstalling)

Usage
-----

Once the package is installed, you should now be able to directly import spynal and
its modules/functions in your code/notebooks in any of the standard Python ways:

.. code-block:: python

    import spynal  
    import spynal as spy  
    import spynal.spectra as spectra  
    from spynal.spectra import spectrogram  