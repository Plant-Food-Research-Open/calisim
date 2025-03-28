Installation
============

The easiest way to install calisim is by using pip:

.. code-block:: bash

    pip install calisim

calisim's default installation will not include all optional dependencies. You may be interested in one or more extras:

.. code-block:: bash

    # Install PyTorch extras
    pip install calisim[torch]

    # Install Hydra extras
    pip install calisim[hydra]

    # Install TorchX extras
    pip install calisim[torchx]

    # Install multiple extras
    pip install calisim[torch,hydra,torchx]

Usage with Docker
-----------------

You may also want to execute calisim inside of a Docker container. You can do so by running the following:

.. code-block:: bash

    # Change the image version as needed
    export CALISIM_VERSION=latest

    # Get docker-compose.yaml file
    wget https://raw.githubusercontent.com/Plant-Food-Research-Open/calisim/refs/heads/main/docker-compose.yaml

    # Pull the image
    docker compose pull calisim

    # Run an example
    docker compose run --rm calisim python examples/optimisation/optuna_example.py
