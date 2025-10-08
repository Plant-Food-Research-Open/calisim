.. figure:: ../assets/calisim_logo.png
   :alt: calisim
   :width: 380px
   :height: 120px

calisim
-------

`PyPI <https://pypi.python.org/pypi/calisim>`_ |
`GitHub <https://github.com/Plant-Food-Research-Open/calisim/>`_ |
`API <https://calisim.readthedocs.io/en/latest/api_reference/index.html>`_ |
`Changelog <https://calisim.readthedocs.io/en/latest/changelogs/changelog.html>`_ |
`Examples <https://github.com/Plant-Food-Research-Open/calisim/tree/main/examples>`_ |
`Releases <https://github.com/Plant-Food-Research-Open/calisim/releases>`_ |
`Docker <https://github.com/Plant-Food-Research-Open/calisim/pkgs/container/calisim>`_ |
`Binder <https://mybinder.org/v2/gh/Plant-Food-Research-Open/calisim.git/HEAD>`_

calisim is an open-source, low-code model calibration library that streamlines and standardises your workflows, while aiming to be as flexible and extensible as needed to support more complex use-cases. Using calisim will speed up your experiment cycle substantially and make you more productive.

calisim is primarily a wrapper around popular libraries and frameworks including Optuna, PyMC, scikit-learn, and emcee among many others. The design and simplicity of calisim was inspired by the scikit-learn and PyCaret libraries.

Features and Functionality
--------------------------

* A standardised and streamlined interface to multiple calibration procedures and libraries.
* A low-code library, allowing modellers to rapidly construct multiple workflows for many calibration procedures.
* An object-oriented programming architecture, allowing users to easily extend and modify calibration workflows for their own complex modelling use-cases.
* An unopinionated approach to working with simulation models, allowing users to calibrate both Python-based and non-Python-based models.
* Optional integration with PyTorch for access to more sophisticated Gaussian Process and deep learning surrogate models, state-of-the-art evolutionary algorithms, and deep generative modelling for simulation-based inference.

Communication
-------------

* `GitHub Discussions <https://github.com/Plant-Food-Research-Open/calisim/discussions>`_ for questions.
* `GitHub issues <https://github.com/Plant-Food-Research-Open/calisim/issues>`_ for bug reports and feature requests.

Contributions and Support
-------------------------

Contributions are more than welcome. For general guidelines on how to contribute to this project, take a look at `CONTRIBUTING.md. <https://github.com/Plant-Food-Research-Open/calisim/blob/main/CONTRIBUTING.md>`_

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   tutorials/basics/installation.rst
   tutorials/basics/quickstart.rst
   api_reference/index.rst
   changelogs/changelog.rst
   licenses/license.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
