# Raw Data

This folder contains the raw data collected for this study. While the `dibco\*.csv` files contain data about the binarization quality, the file `bin_times.csv` contains data about the binarization execution time per image.

## Binarization Quality

For each of the 7 folds in this study, one model was trained, for each evaluated network configuration, on five datasets, while one dataset was used for validation and one for testing. 
The file names state the name of the test dataset, which was binarized using its respective models. 
The binarization quality was assessed using the tool provided by the organizers of the DIBCO/H-DIBCO competitions, which can be found [here](http://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO_metrics.zip).

As mentioned in the description of the `binarize.py` script, the different network configurations with respect to footprint size, scale factor and loss function are represented by following mode strings:

| Mode String        | Footprint Size           | Scale Factor  | Loss Function |
| ------------- |-------------:| -----:| :-------:|
| dyn_wgt_sc2_16_201e      | 4 | 2 | dynamic |
| dyn_wgt_sc2_4equal_201e |  2 | 2 | dynamic |
| dyn_wgt_sc2_64_201e |  8 | 2 | dynamic |
| dyn_wgt_sc1_16_201e      | 4 | 1 | dynamic |
| dyn_wgt_sc4_16_201e |  4 | 4 | dynamic |
| no_wgt_sc2_16_201e      | 4 | 2 | no weight |
| stat_wgt20_sc2_16_201e     | 4 | 2 | static weight (2.0) |
| stat_wgt05_sc2_16_201e     | 4 | 2 | static weight (0.5) |

## Execution Performance

The csv file containing the execution times lists for each image in the respective test dataset how long it took to binarize this particular image for the three evaluated configurations of different footprint sizes.
The execution times are reported in seconds.
