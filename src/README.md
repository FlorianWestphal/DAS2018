# Implementation

In order to train a model for binarization or to binarize images using a trained model, the image collection to be used needs to be converted into a database file first, which will then be used by the implementation for training or binarizing images.


## Database Creation

The database can be created using the `create_database.py` script by first modifying the configuration in `binarization/config.py` and then running the script as follows:

<code shell>
./create_database.py ../test.hdfs
</code>

This generates a database file, called `test.hdfs`, which contains all configured datasets, formatted with the configured scale factor and footprint size.

The scale factor can be changed by setting `self.scale` in `binarization/config.py`, while the footprint size is modified by setting `self.inputs`. Note that inputs is footprint size * footprint size.

The datasets are configured by modifying the `self.datasets` dictionary. The dataset name is entered as dictionary key and the paths to the folder containing the original dataset images, the folder containing the dataset's ground truth images and the folder containing the corresponding weight files are entered in the list of strings belonging to this key. Note that the file names of the original and the ground truth images need to be the same, apart from the file ending. The weight files need to be generated from the ground truth using the tool provided by the organizers of the DIBCO and H-DIBCO competitions, which can be found [here](http://vc.ee.duth.gr/h-dibco2016/benchmark/BinEvalWeights.zip). For example, if the original file is called `H01.bmp`, then the ground truth file should be called `H01.tif` or `H01.bmp` and the weight files should be called `H01_PWeights.dat` and `H01_RWeights.dat`.

## Training

In order to train a model for binarization, one has to create a database file, as described above and provide a configuration file to the `train.py` script as follows:

<code shell>
./train.py ../test.config 42 # configuration file and set random seed
</code>

The configuration file should look as follows:

```shell
[config]

# Network Config
input-factor: 5
# for using static weight, set this option to 0.5 or 2.0 and the next option to False
# for using no weight, set this option to 1.0 and the next option to False
positive-weight:
use-dynamic-weights: True

# Run Config
epochs: 201
hdfs-path: "/path/to/database/file/test.hdfs"
# configure from where model should be loaded (new model if empty)
load-path:
# configure where model should be stored
store-path: "/model/store/path/test_run"

# Train Config
# dataset names, as they can be found in the database file
train: dibco09,dibco10,dibco11,dibco12,dibco13,dibco14
test: dibco16
cluster-path:
# configure which of the train datasets should be used for validation
fold-number: 6
fold-seed: 42
fold: 0

# Bin Config
batch-size: 32
sample-size: 2048
sample-seed: 42
log-path: "/path/to/tensorflow/log/test_run"

[random]

ratio: 0.2
seed: 42
```

## Binarization

Using a trained model, the images of a dataset contained in the database file can be binarized using the `binarize.py` script as follows:

<code shell>
# ./binarize.py <dataset name> <model path> <target path> <mode string> 
./binarize.py dibco16 ../model/dyn_wgt_sc2_16_201e_best_err ../result/ dyn_wgt_sc2_16_201e
</code>

The command above will binarize all images in the `dibco16` dataset using the specified model and store the images in the folder `../result/`. The used network configuration with respect to footprint size, scale factor and used loss function is specified with the provided mode string and should match the configuration, which was used to train the provided model.

The supported mode strings and their corresponding configurations are as follows:

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

Note that the paths to the database file to use need to be adjusted either in `binarization/config.py` or in the `binarize.py` script. This can be done by changing the value of `self.hdfs_path` in `binarization/config.py` or of `config.hdfs_path` in `binarize.py` appropriately.
