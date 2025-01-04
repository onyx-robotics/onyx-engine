.. _installation:

Installation
===============

Welcome to the Onyx Engine! To quickly get started, follow these steps:

#. **Create an Account**

   Create your account on the `Engine Platform <https://engine.onyx-robotics.com>`_.

#. **Set Your API Key**

    You can find your API key in the `Engine Platform <https://engine.onyx-robotics.com>`_ account settings.

    Copy your API key and add it to your environment variables:

    .. code-block:: bash

        echo 'export ONYX_API_KEY="YOUR_API_KEY"' >> ~/.bashrc
        source ~/.bashrc

    .. warning::
        
        Do not share your API key with anyone. It is used to authenticate your requests to the Engine.
        You can reset your API key at any time in the `Engine Platform <https://engine.onyx-robotics.com>`_ account settings.

#. **Install the Engine API Python Package**

    Install the onyx-engine package using pip:
    
    .. code-block:: bash
    
        pip install onyxengine

    You can test your installation by retrieving the metadata of the example training dataset in the Engine.

	.. code-block:: python

		import json
		import onyxengine as onyx

		metadata = onyx.get_object_metadata('example_train_data')
		print(json.dumps(metadata, indent=2))

	The output should look something like this:

	.. code-block:: text

		{
		  "name": "example_train_data",
		  "object_type": "dataset",
		  "object_config": {
			"features": [
			  "acceleration",
			  "velocity",
			  "position",
			  "control_input"
			],
			"num_outputs": 1,
			"num_state": 2,
			"num_control": 1,
			"dt": 0.0025
		  },
		  "status": "active",
		  "owner": "Ted Lutkus",
		  "last_updated": "2025-01-01T14:22:34.427448+00:00",
		  "date_created": "2025-01-01T14:22:34.427448+00:00",
		  "version": 1
		}

    If you see the metadata of the example dataset, your installation was successful! Now you're ready to start :ref:`training-models`. 