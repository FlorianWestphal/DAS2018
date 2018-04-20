import ConfigParser

import numpy as np
import random

class BaseConfig(object):
    CONFIG = 'config'
    PARAM_FMT = '\t{}: {}\n'
    HEAD_FMT = '{}:\n'

    def __init__(self, config_path):
        self._config = ConfigParser.RawConfigParser(allow_no_value=True)
        self._config.read(config_path)

    def _parse_hdfs(self, config):
        hdfs_path = self._parse_string(config, self.HDFS)
        if hdfs_path is None:
            raise ValueError("Database file must be configured!")
        return hdfs_path

    def _parse_load(self, config):
        load_path = self._parse_string(config, self.LOAD)
        if load_path is None:
            raise ValueError("Model load path must be configured!")
        return load_path

    def _parse_int(self, config, name):
        return self._parse_int_from(config, self.CONFIG, name)

    def _parse_int_from(self, config, section, name):
        result = None
        entry = config.get(section, name)
        if len(entry) != 0:
            result = int(entry)
        
        return result
   
    def _parse_float(self, config, name):
        return self._parse_float_from(config, self.CONFIG, name)
 
    def _parse_float_from(self, config, section, name):
        result = None
        entry = config.get(section, name)
        if len(entry) != 0:
            result = float(entry)
        return result

    def _parse_bool(self, config, name):
        result = False
        entry = config.get(self.CONFIG, name)
        if len(entry) != 0:
            result = (entry.lower() == 'true')
        return result

    def _parse_list_from(self, config, section, name):
        result = []
        entry = config.get(section, name)
        if len(entry) != 0:
            entries = entry.split(',')
            for e in entries:
                result.append(e.strip())
        return result

    def _parse_string(self, config, name):
        result = config.get(self.CONFIG, name)
        if len(result) == 0:
            result = None
        else:
            result = result.replace('"', '').replace("'", "")
        return result

    def _parse_option(self, config, option_map):
        """Parse the given option as either true or false value.

        Keyword arguments:
        config      --  the parsed configuration
        option_map  --  map containing the name of the option as key and a
                        list of values, where the first value is the value,
                        whose presence results in true to be returned
        
        Returns:
        true -- if the given option was set to the first value in the value list
        """
        val = self._parse_string(config, option_map['key'])
        result = True
        if val is not None:
            result = (val == option_map['val'][0])
        return result

    def __str__(self):
        return 'Configuration\n----\n\n'

class NetworkConfig(BaseConfig):
    # network configuration
    INPUT_FACT = 'input-factor'
    POS_WEIGHT = 'positive-weight'
    DYN_WEIGHT = 'use-dynamic-weights'

    def __init__(self, config_path):
        super(NetworkConfig, self).__init__(config_path)

        # parse network configuration
        self.input_factor = self._parse_int(self._config, self.INPUT_FACT)
        self.positive_weight = self._parse_float(self._config, self.POS_WEIGHT)
        self.dynamic_weights = self._parse_bool(self._config, self.DYN_WEIGHT)

    def __str__(self):
        desc = super(NetworkConfig, self).__str__()
        desc += self.HEAD_FMT.format('Network Configuration')
        desc += self.PARAM_FMT.format(self.INPUT_FACT, self.input_factor)
        desc += self.PARAM_FMT.format(self.POS_WEIGHT, self.positive_weight)
        desc += self.PARAM_FMT.format(self.DYN_WEIGHT, self.dynamic_weights)
        desc += '\n'

        return desc

class RunConfig(BaseConfig):
    EPOCHS = 'epochs'
    HDFS = 'hdfs-path'
    LOAD = 'load-path'
    STORE = 'store-path'

    def __init__(self, config_path):
        super(RunConfig, self).__init__(config_path)
        
        # parse general run configuration
        self.epochs = self._parse_int(self._config, self.EPOCHS)
        self.hdfs_path = self._parse_hdfs(self._config)
        self.load_path = self._parse_load(self._config)
	self.store_path = self._parse_string(self._config, self.STORE)

    def __str__(self):
        desc = super(RunConfig, self).__str__()
        desc += self.HEAD_FMT.format('Run Configuration')
        desc += self.PARAM_FMT.format(self.EPOCHS, self.epochs)
        desc += self.PARAM_FMT.format(self.HDFS, self.hdfs_path)
        desc += self.PARAM_FMT.format(self.LOAD, self.load_path)
        desc += self.PARAM_FMT.format(self.STORE, self.store_path)
        desc += '\n'

        return desc

class TrainConfig(BaseConfig):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'
    CLUSTER = 'cluster-path'
    FOLD_NUM = 'fold-number'
    FOLD_SEED = 'fold-seed'
    FOLD = 'fold'

    def __init__(self, config_path):
        super(TrainConfig, self).__init__(config_path)
        
        self._cluster_path = self._parse_string(self._config, self.CLUSTER)
        self._train = self._parse_list_from(self._config, self.CONFIG,
                                                            self.TRAIN)
        self._test = self._parse_list_from(self._config, self.CONFIG, self.TEST)
        self._valid = self._parse_list_from(self._config, self.CONFIG,
                                                                self.VALID)
        self._fold_num = self._parse_int(self._config, self.FOLD_NUM)
        self._fold_seed = self._parse_int(self._config, self.FOLD_SEED)
        self._fold = self._parse_int(self._config, self.FOLD)

    def _read_clusters_from(self, file_path):
        table = np.genfromtxt(file_path, delimiter=',', dtype=None)
        clusters = {}
        for entry in table:
            cluster = str(entry[-1])
            if not clusters.has_key(cluster):
                clusters[cluster] = []
            clusters[cluster].append((entry[0].split('.')[0], entry[1]))
        return clusters

    def configured_datasets(self):
        train = []
        test = []
        valid = None
        if self._cluster_path is None:
            train = self._train
            test = self._test
        else:
            clusters = self._read_clusters_from(self._cluster_path)
            for cluster in self._train:
                train += clusters[cluster]
            for cluster in self._test:
                test += clusters[cluster]

        if len(self._valid) > 0:
            valid = self._valid
        elif self._fold_num is not None:
            random.seed(self._fold_seed)
            random.shuffle(train)
            tmp = [train[i::self._fold_num] for i in xrange(self._fold_num)]
            valid = tmp.pop(self._fold)
            train = [item for sublist in tmp for item in sublist]

        return train, valid, test

    def __str__(self):
        desc = super(TrainConfig, self).__str__()
        desc += self.HEAD_FMT.format('Train Configuration')
        desc += self.PARAM_FMT.format(self.TRAIN, self._train)
        desc += self.PARAM_FMT.format(self.TEST, self._test)
        desc += self.PARAM_FMT.format(self.VALID, self._valid)
        desc += self.PARAM_FMT.format(self.CLUSTER, self._cluster_path)
        desc += self.PARAM_FMT.format(self.FOLD_NUM, self._fold_num)
        desc += self.PARAM_FMT.format(self.FOLD_SEED, self._fold_seed)
        desc += self.PARAM_FMT.format(self.FOLD, self._fold)
        desc += '\n'

        return desc

class BinConfig(NetworkConfig, RunConfig, TrainConfig):
    BATCHES = 'batch-size'
    SSIZE = 'sample-size'
    SSEED = 'sample-seed'
    LOG_PATH = 'log-path'

    def __init__(self, config_path):
        super(BinConfig, self).__init__(config_path)
        self.batches = self._parse_int(self._config, self.BATCHES)
        self.sample_size = self._parse_int(self._config, self.SSIZE)
        self.sample_seed = self._parse_int(self._config, self.SSEED)
        self.log_path = self._parse_string(self._config, self.LOG_PATH)

    def _parse_load(self, config):
        """Parse path to trained model to load. The path is allowed to be
        empty."""
        return self._parse_string(config, self.LOAD)

    def __str__(self):
        desc = super(BinConfig, self).__str__()
        desc += self.HEAD_FMT.format('Binarization Configuration')
        desc += self.PARAM_FMT.format(self.BATCHES, self.batches)
        desc += self.PARAM_FMT.format(self.SSIZE, self.sample_size)
        desc += self.PARAM_FMT.format(self.SSEED, self.sample_seed)
        desc += self.PARAM_FMT.format(self.LOG_PATH, self.log_path)
        desc += '\n'

        return desc

