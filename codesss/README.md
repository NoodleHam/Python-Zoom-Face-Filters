### 1. Facial Landmark Detection
Run `python3 1-facial-landmark-original.py` to visualize facial landmarks using functions provided by OpenCV.
Press "q" to exit. (Credit to [Daniel Otulagun](https://github.com/Danotsonof/))

### 2. Segmenting the Landmarks
Run `python3 2-facial-landmark-segmentation.py` to visualize facial landmarks with contours around each region.
(Credit to [Adrian Rosebrock](https://github.com/jrosebr1))

### 3. Detecting Gaze
I hand-made a convolutional filter to detect the location of the pupil in an eye. To visualize
the feature maps and the estimated pupil location, run\
`python3 4-visualize-gaze.py`

### 4. Animating the Eyes
With this information, we can animate a set of cartoon eyes to look in the direction
you're looking. This is the final product. To see this, run
`python3 codesss/5-cartoon-eyes.py`.

To see the demo outputs of each script, visit Tech@NYU's Events page at
[https://events.techatnyu.org/past-events/diy-zoom-filters-0324](https://events.techatnyu.org/past-events/diy-zoom-filters-0324).