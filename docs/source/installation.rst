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
		"type": "dataset",
		"created_at": "2025-04-14T15:54:40.516003+00:00",
		"config": {
			"type": "dataset",
			"features": [
			"acceleration_predicted",
			"velocity",
			"position",
			"control_input"
			],
			"dt": 0.0025
		},
		"status": "active",
		"team_id": "7e6c03eb-217e-41d6-82d2-4b5d6109694f",
		"id": "52aea6f3-f61e-487b-981b-901e11b4a9c0",
		"user_id": "e28647cf-b254-483b-ac59-1db7668f7f03",
		"type_metadata": {
			"id": "52aea6f3-f61e-487b-981b-901e11b4a9c0",
			"num_points": 960800,
			"memory_bytes": 71807741
		}
		}

    If you see the metadata of the example dataset, your installation was successful! Now you're ready to start :ref:`training-models`. 