# Training Models

## Training an AI model

```{figure} _static/model_features.png
:alt: Model Features
:align: center
:width: 100%
``` 
Once a dataset has been processed successfully, AI models can be trained from it. There are multiple ways to create a new model from a dataset.

* In the System Overview, hover over a dataset node until a plus button appears on the right side. Click the plus button to open the menu, and click “Train Model”  
* Click on a dataset node to open the Inspector. Under Actions, click “Train Model”.  
* In the System Overview, click on the name of a dataset to open the detailed object view. Click the “Train Model” model button in the top right corner. This button is available for active (successfully processed) datasets.

Clicking “Train Model” will now create a new model object. This object is a child of the dataset it was created from and will appear as such in the System Overview and lineage.

## Model Features and Configuration

```{figure} _static/model_config.png
:alt: Model Config
:align: center
:width: 100%
``` 
Newly created models will have a preset configuration that is ready for training. The first feature of a dataset is used as the output and all other features are used as the input.

You can change the input and output features via the Features view of the Training tab. Click the plus button to add a new input/output. Use the dropdown menu to select a feature. Click the settings icon to access Scale Configuration details for a feature.

You can modify the advanced configuration settings of the model by switching to the Configuration view of the Training Tab. From here, the model type, model configuration, training configuration, optimizer config, and learning rate scheduler config are accessible.

Once you are satisfied with the configuration, click “Run Training” to start a new model training job.

## Jobs

```{figure} _static/jobs.png
:alt: Jobs
:align: center
:width: 100%
``` 
All trained and optimized model versions of a model are viewable from the “Jobs” tab in the detailed object view.

Training jobs that are still in progress will have the status “Training”. Training jobs that are completed will have the status “Completed”.

Click on the “Expand” arrow to view the training config for a job.