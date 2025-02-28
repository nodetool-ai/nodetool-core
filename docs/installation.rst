.. highlight:: shell

============
Installation
============


Stable release
-------------

To install Nodetool Core, run this command in your terminal:

.. code-block:: console

    $ pip install nodetool-core

This is the preferred method to install Nodetool Core, as it will always install the most recent stable release.

From sources
-----------

The sources for Nodetool Core can be downloaded from the repository.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/username/nodetool-core

Or download the tarball:

.. code-block:: console

    $ curl -OJL https://github.com/username/nodetool-core/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


Development Installation
-----------------------

For development, you can install the package in development mode:

.. code-block:: console

    $ pip install -e .
    $ pip install -r requirements-dev.txt

Or use the Makefile:

.. code-block:: console

    $ make dev-install 