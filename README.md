This dataloader was coded over the course of 3 months before the scope of my assignment was changed without warning.
It was coded completely independently and with little to no assistance.

The purpose of the dataloader was to parse through SAR (Synthetic Apeture Radar) datasets that were stored in HDF5 format. The dataloader has the following methods: converting labels from tensors to ints, creating a label mapping dictionary, ensuring image data has the correct dimensions, labels have the same case, and correcting label numbers in a combined global dictionary.

This was done to make further data processing and experiments easier instead of loading data manually each time a senior employee needed to work with it.

I learned a lot about object oriented programming concepts while constructing this script, and managed to acheive an accuracy of 90% in a simple classification task when using this script.
