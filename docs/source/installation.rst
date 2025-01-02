.. _installation:

Installation
===============

To begin using the Onyx Engine, follow these steps:

#. **Create an Account**

   Create your account on the `Engine Platform <https://engine.onyx-robotics.com>`_.

#. **Set Your API Key**

    You can find your API key in the `Engine Platform <https://engine.onyx-robotics.com>`_ account settings.

    Copy your API key and add it to your environment variables:

    .. code-block:: bash

        echo 'export ONYX_API_KEY="YOUR_API_KEY"' >> ~/.bashrc
        source ~/.bashrc

    .. warning::
        
        Do not share your API key with anyone. It is used to authenticate your requests to the Engine Platform.
        You can reset your API key at any time in the `Engine Platform <https://engine.onyx-robotics.com>`_ account settings.

#. **Install the Engine API Python Package**

    Install the onyx-engine package using pip:
    
    .. code-block:: bash
    
        pip install onyxengine

    You can test your installation by retrieving the metadata of the example dataset in the Engine.

    .. code-block:: python

        import onyxengine as onyx

        metadata = onyx.get_object_metadata('example_dataset')
        print(metadata)

    If you see the metadata of the example dataset, your installation was successful! Now you're ready to start :ref:`training-models`. 