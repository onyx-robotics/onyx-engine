# Uploading a dataset

Uploading a dataset is the first important step towards solving real-world accuracy problems for your team’s physics models.

```{figure} _static/upload.png
:alt: Upload
:align: center
:width: 100%
```

You can upload a dataset into the Onyx Engine via the following methods:

1. Click the “Upload” button from the System Overview  
2. Click the “Upload” button from the System Table  
3. Drag and drop a folder/file directly into the System Overview or System Table.

The Upload Dataset window provides an example dataset for reference.

## Collecting Data

### What kind of data do I need to collect?

A timeseries of \[predicted\_outputs, system\_states, control\_inputs\]. Often the predicted output is just a copy of the system state.

### How much data do I need?

Our model training is highly data-efficient and typically needs less than one hour of data collection at a modest sampling rate (\~50-400Hz).

### Can I upload multiple files?

Yes. If your dataset is broken up into chunks/episodes as files, our processing pipeline will concatenate them vertically into one long timeseries as a csv. Please let us know if you have questions or need help.

## Uploading

Once collected, upload your files directly into the window and click “Upload”. Once the upload has completed, you will be taken directly into the detailed object view of your new dataset.