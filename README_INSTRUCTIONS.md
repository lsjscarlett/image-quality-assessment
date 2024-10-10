# Local Business Image Quality Assessment

This project evaluates the quality of images from local businesses using a pre-trained model. The results can help businesses understand and improve the visual appeal of their images.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Examples](#examples)
5. [Results](#results)

## Introduction

This project uses a CNN deep learning model to assess the quality of images. The model scores images based on aesthetic and technical quality. The scores can be used to identify which images are most visually appealing or require improvement.

## Setup

To set up the project, follow these steps:

1. ### Prerequisites

Ensure you have the following installed:
- Python 3.x
- ImageMagick 7.x (for handling HEIC and other image formats)

Navigate to the Directory Where You Want to Clone the Repository:
```bash
cd /path/to/your/directory
```

2. ### Clone the project from GitHub
```bash
git clone https://github.com/lsjscarlett/image-quality-assessment
```

3. ### Install required packages
```bash
 pip install nose
 pip install scikit-learn
 pip install pillow
 pip install utilis
 pip install tensorflow
 pip install numpy
 brew install libheif
 pip install pyheif pillow
 brew install imagemagick

```

4. ### Code Modifications
Modify predict.py file from utilis folder
```
from src.utils.utils import calc_mean_score, save_json
from src.handlers.model_builder import Nima
from src.handlers.data_generator import TestDataGenerator
```

Modify model_builder.py from handles folder
```
from src.utils.losses import earth_movers_distance
```

Make change to data_generator.py from handles folder
```
from src.utils import utils
```

Modify the predict function in predict.py file from evaluator folder
```
def predict(model, data_generator):
    return model.predict(data_generator, verbose=1)
```
## Usage
Get score for a single image file
```bash
PYTHONPATH=$(pwd) python src/evaluater/predict.py \
  --base-model-name MobileNet \
  --weights-file $(pwd)/models/MobileNet/weights_mobilenet_technical_0.11.hdf5 \
  --image-source $(pwd)/src/tests/test_images/42039.jpg
```

Get score for a batch of image files
```bash
PYTHONPATH=$(pwd) python src/evaluater/predict.py \
  --base-model-name MobileNet \
  --weights-file $(pwd)/models/MobileNet/weights_mobilenet_technical_0.11.hdf5 \
  --image-source $(pwd)/src/tests/test_images/restaurants/
```

## Examples
### Local Restaurants
#### Yard House
![Yard House](src/tests/test_images/restaurants/retaurant_1_yard_house.jpg)
#### Water Tower Kitchen
![Water Tower Kitchen](src/tests/test_images/restaurants/restuarant_2_water_tower_kitchen.jpg)
#### Be Steak A
![Be Steak A](src/tests/test_images/restaurants/restaurant_3_the_table.jpg)
#### ABVE the Basics
![ABVE the Basics](src/tests/test_images/restaurants/restaurant_4_be_steak_a.jpg)
#### The Table
![The Table](src/tests/test_images/restaurants/restaurant_5_abve_the_basics.jpg)

### Local Retails
#### Bea Bark and Moss
![Bea Bark and Moss](src/tests/test_images/retailer/retail_1_bea_bark_and_moss.jpg)
#### MME
![MME](src/tests/test_images/retailer/retail_2_mme.jpg)
#### Button Down
![Button Down](src/tests/test_images/retailer/retail_3_button_down.jpg)
#### Hammer and Levis
![Hammer and Levis](src/tests/test_images/retailer/retail_4_hammer_and_levis.jpg)
#### Redemption
![Redemption](src/tests/test_images/retailer/retail_5_redemption.jpg)

### Local Law Firms
#### My Personal Injury Lawyers
![My Personal Injury Lawyers](src/tests/test_images/law_firms/law_firm_1_my_personal_injury_lawyers.png)
#### RMD Law
![RMD Law](src/tests/test_images/law_firms/law_firm_2_rmd_law.png)
#### Solution Now
![Solution Now](src/tests/test_images/law_firms/law_firm_3_solution_now.png)
#### Deldar Legal
![Deldar Legal](src/tests/test_images/law_firms/law_firm_4_deldar_legal.jpg)
#### The Law Collective
![The Law Collective](src/tests/test_images/law_firms/law_firm_5_the_law_collective.jpg)

### Local CPA Firms
#### Dimov Tax Services
![Dimov Tax Services](src/tests/test_images/cpa_firms/cpa_1_dimov_tax_services.jpg)
#### Fugate Business Solutions
![Fugate Business Solutions](src/tests/test_images/cpa_firms/cpa_2_fugate_business_solutions.png)
#### SF Bay Tax
![SF Bay Tax](src/tests/test_images/cpa_firms/cpa_3_sfbaytax.png)
#### Tax Relief USA
![Tax Relief USA](src/tests/test_images/cpa_firms/cpa_4_tax_relief_usa.png)
#### Advantum Tax
![Advantum Tax](src/tests/test_images/cpa_firms/cpa_5_advantum_tax.png)

### Local Hair Salons
#### Her Studio Hair Salon
![Her Studio Hair Salon](src/tests/test_images/hair_salons/salon_1_Her_Studio_Hair_Salon.jpg)
#### The Studio Los Gatos
![The Studio Los Gatos](src/tests/test_images/hair_salons/salon_2_The_Studio_Los_Gatos.jpg)
#### Essence Salon
![Essence Salon](src/tests/test_images/hair_salons/salon_3_Essence_Salon.jpg)
#### Limon Salon
![Limon Salon](src/tests/test_images/hair_salons/salon_4_Limon_Salon.jpg)
#### My Stylist Salon
![My Stylist Salon](src/tests/test_images/hair_salons/salon_5_My_Stylist_Salon.jpg)

## Results
### Restaurant Result
![Restaurant_result](src/tests/test_images/Results/restaurant_result.png)
### Retail Results
![Restaurant_result](src/tests/test_images/Results/retail_result.png)
### Law Firm Result
![Law_firm_result](src/tests/test_images/Results/law_firm_result.png)
### CPA Firm Result
![Restaurant_result](src/tests/test_images/Results/CPA_firm_result.png)
### Hair Salon Result
![Hair_salon_result](src/tests/test_images/Results/hair_salon_result.png)
