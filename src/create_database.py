import sys
import ConfigParser
import numpy as np

import binarization


if len(sys.argv) != 2 and len(sys.argv) != 4:
    print "usage:", sys.argv[0], "<target> [<dataset> <path>]"
    exit(1)

config = binarization.config.Configuration()
conv = binarization.converter.ToSeqConverter(config)

with binarization.storage.StorageWriter(sys.argv[1], conv) as store:
    datasets = config.datasets
    for dataset in datasets:
        store.store_dataset(dataset, datasets[dataset][0], 
                            datasets[dataset][1], datasets[dataset][2])
    if len(sys.argv) == 5:
        store.store_image(sys.argv[2], sys.argv[3])
