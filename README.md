# GridEdgeDetector

## Install

Clone this repository

```git clone https://github.com/philipp-schoennenbeck/GridEdgeDetector```

Move into repository
```cd GridEdgeDetector```

Create a new python environment using conda

```conda create -n ged```

Activate conda environment
```conda activate ged```

Run setup.py file

```pip install -e .```

Now you can start the GUI with 
```ged```

## Parameters

Pixel spacing: The resolution of your image in Å/pixel. Automatically extracted when using mrc-files.

Grid hole size: The diameter of your grid hole in μm. It is usually 2 or 1.2.

Threshold: The value to distinguish between finding a hole and not finding a hole. Look at the graph to get a better understanding about this value for your data.

Distance: How far inside the grid hole the mask should extend in Å. Positive values create a ring link mask. Negative a smaller circle than the grid hole.

Return ring width: Create the mask as a ring around the grid hole with this width in Å.

Outside circle coverage: How much of the image is allowed to be outside of the grid hole as a decimal value between 0-1.

Inside circle coverage: How much of the image has to be inside of the grid hole as a decimal value between 0-1.

High pass filter: Apply a high pass filter of this size before searching for the edge. In Å. 


Resize to: The resolution the image is resized to to speed up the algorithm in Å/pixel.

Detect ring: Whether to compare the values of a circle or a ring to every other value. A ring detection can be useful for high pass filtered images.

Ring width: If Detect ring is active this is the parameter for the ring width in Å.

CPUs to use: Number of CPUs to use for parallization.

Wobble: A small value to find more accurate results. THe algorithm finds first the best coordinate for the circle and then changed the circle size and the coordinate by this amount to find slightly better results. 0-1

Crop: The amount of Å to crop the iamge before searching for the grid edge. Sometimes image artifacts on the image edges can influence the algorithm. 