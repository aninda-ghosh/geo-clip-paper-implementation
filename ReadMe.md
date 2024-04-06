# GEOCLIP 

Paper Implementation with Pytorch Lightning Support for easy training and prediction scripts with multi-gpu support.

Paper Link: https://arxiv.org/pdf/2309.16020.pdf

## Description

Worldwide Geo-localization aims to pinpoint the precise location of images taken anywhere on Earth. This task has considerable challenges due to the immense variation in geographic landscapes. The image-to-image retrieval-based approaches fail to solve this problem on a global scale as it is not feasible to construct a large gallery of images covering the entire world. Instead, existing approaches divide the globe into discrete geographic cells, transforming the problem into a classification task. However, their performance is limited by the predefined classes and often results in inaccurate localizations when an image’s location significantly deviates from its class center. To overcome these limitations, we propose GeoCLIP, a novel CLIP-inspired Image-to-GPS retrieval approach that enforces alignment between the image and its corresponding GPS locations. GeoCLIP’s location encoder models the Earth as a continuous function by employing positional encoding through random Fourier features and constructing a hierarchical representation that captures information at varying resolutions to yield a semantically rich high-dimensional feature suitable to use even beyond geo-localization. To the best of our knowledge, this is the first work employing GPS encoding for geo-localization. We demonstrate the efficacy of our method via extensive experiments and ablations on benchmark datasets. We achieve competitive performance with just 20% of training data, highlighting its effectiveness even in limited-data settings. Furthermore, we qualitatively demonstrate geo-localization using a text query by leveraging the CLIP backbone of our image encoder. The project webpage is available at: https://vicentevivan.github.io/GeoCLIP

## Changelogs

### v1.0.0

- Organized the code in a structure with easy training and testing scripts.
- Added support for pytorch lightning and tensorboard logging.
- Added a config file to make experiments more scalable.

### v1.0.1

- Added evaluation script
- Performed evaluation on YFCC4K dataset.
    `Accuracy at 2500 km: 67.77`
    `Accuracy at 750 km: 44.44`
    `Accuracy at 200 km: 21.94`
    `Accuracy at 25 km: 9.08`
    `Accuracy at 1 km: 1.21`
- Added script to predict location for a given image.

## Test Results

### Test 1 (ASU Library)

#### Fed Image

![](test_image\asu_library.jpg)

#### Results

- Image 1 GPS (top 3):
    ` tensor([[  33.4187, -111.9389], [  33.4187, -111.9369], [  33.4192, -111.9505]]) `
- Image 1 Probability: 
    `tensor([0.0082, 0.0080, 0.0079])`

#### Location from Google Maps

![](test_image\asu_library_location.png)

### Test 2 (Eiffel Tower)

#### Fed Image

![](test_image\eiffel_tower.jpeg)

#### Results

- Image GPS (top 3):
    `tensor([[48.8616,  2.2941], [48.8629,  2.2959], [48.8619,  2.2903]])`
- Image Probability: 
    `tensor([0.0013, 0.0013, 0.0013])`

#### Location from Google Maps

![](test_image\eiffel_tower_location.png)

### Test 3 (Taj Mahal)

#### Fed Image

![](test_image\taj_mahal.jpg)

#### Results

- Image GPS (top 3):
    `tensor([[27.1676, 78.0369], [27.1692, 78.0422], [27.1711, 78.0407]])`
- Image Probability: 
    `tensor([0.0143, 0.0136, 0.0135])`

#### Location from Google Maps

![](test_image\taj_mahal_location.png)