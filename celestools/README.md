Introduction to Celestools
**************************

This Python package contains modules useful for celestial mechanics and in particular for orbital flight. 

Last update: 2020/04/22 by Alexis Petit, Delphine Ly, Gabriel Magnaval.

* Version 1.0.0: Initial version from CelestialPy (constants, coordinates, time scales, TLE).
* Version 1.0.1: Add coordinate and frame transformations using Orekit.
* Version 1.0.2: Determination of the satellite coordinate systems RSW and NWT as defined by Vallado, 2013, p.157. 
* Version 1.0.3: Modification of the keys.
* Version 1.0.4: Vectorization of the conversion modules. Add arguments in the TLE reader.
* Version 1.0.5: Reference frame transformation (with Orekit and Astropy).
* Version 1.0.6: Computation of the collision probability.

Installation 
************

From the repo

.. code:: bash

    $ git clone ssh://git@SMS2:/srv/git/celestools.git

Install the Celestools Python package (setup.py in the package directory)  

.. code:: bash

    $ python setup.py install

To remove 

.. code:: bash

    $ pip uninstall

Benchmark
*********

In the directory *tests*, you check the package with

.. code:: bash

    $ python vallado.py
