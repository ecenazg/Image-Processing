# Image Proccesing

- The project provides the user to experiment with different values for the sigma parameter of the Gaussian filter and unsharp masking techniques.

- The background image is smoothed using the Gaussian filter, which eliminates noise. To determine the ideal proportion of smoothing for their background image, the user can experiment with various sigma values. As the sigma value increases, the applied image becomes even more smoothed.

- The foreground object is sharpened using the unsharp masking method, which brings out its details. The user can change sigma values according to their wishes like gaussian smoothing method. As the sigma value increases, the details of the applied image become sharper.

- The last step of the process is to combine the sharpened foreground object and the flattened background image. The project achieves assignment by copying the pixels of the foreground object to the corresponding pixels in the background image, where the pixels are in the bounding box generated by the GrabCut algorithm.

## Results 

1.	Creating a binary mask of the foreground object using the GrabCut algorithm
2.	Smoothing the background image using a Gaussian filter with a given sigma values which are sigma 5 and sigma 10.
3.	Sharpening the foreground object using an unsharp masking technique with a given sigma values which are sigma 3 and sigma 8.
4.	Combining the sharpened foreground object and smoothed background image into a final image
5.	I used two example images to test this process: a foreground image of a dog and a background image of a landscape.
