


class Configuration(object):

    def __init__(self):
        self._conversion_config()
        self._network_config()
        self._run_config()
        self._dataset_config()

    def _conversion_config(self):
        self.patch_size = 64
        self.inputs = 16
        self.preprocess = True
        self.equalize = False
        self.scale = 2

    def _network_config(self):
        self.input_factor = 5
        self.positive_weight = None#2.0
        self.dynamic_weights = True

    def _run_config(self):
        self.epochs = 201
        self.hdfs_path = "/opt/dibco17/dibco17.hdfs"
        self.load_path = None
        self.store_path = "/opt/dibco17/models"
        self.batches = 7
        self.sample_size = 1064
        self.sample_seed = 42
        self.log_path = "/opt/dibco17/log"

    def _dataset_config(self):
        self.datasets = {'dibco09':
                            ['/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco09/org/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco09/gt/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/binarization/recurrent_binarization/weights/dibco09/'],
                        'dibco10':
                            ['/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco10/org/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco10/gt/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/binarization/recurrent_binarization/weights/dibco10/'],
                        'dibco11':
                            ['/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco11/org/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco11/gt/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/binarization/recurrent_binarization/weights/dibco11/'],
                        'dibco12':
                            ['/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco12/org/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco12/gt/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/binarization/recurrent_binarization/weights/dibco12/'],
                        'dibco13':
                            ['/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco13/org/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco13/gt/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/binarization/recurrent_binarization/weights/dibco13/'],
                        'dibco14':
                            ['/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco14/org/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco14/gt/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/binarization/recurrent_binarization/weights/dibco14/'],
                        'dibco16':
                            ['/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco16/org/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/initial_system/dataset/dibco/dibco16/gt/',
                            '/Users/flw/Documents/phd/research/binarization/interactive/binarization/recurrent_binarization/weights/dibco16/']}

    def __str__(self):
        desc = 'Conversion:\n'
        desc += 'Patch Size: {}\n'.format(self.patch_size)
        desc += 'Inputs: {}\n'.format(self.inputs)
        desc += 'Preprocess: {}\n'.format(self.preprocess)
        desc += 'Equalize: {}\n'.format(self.equalize)
        desc += 'Scale: {}\n'.format(self.scale)
        
        desc += '\nNetwork:\n'
        desc += 'Input Factor: {}\n'.format(self.input_factor)
        desc += 'Positive Weight: {}\n'.format(self.positive_weight)

        desc += '\nRun:\n'
        desc += 'Epochs: {}\n'.format(self.epochs)
        desc += 'HDFS Path: {}\n'.format(self.hdfs_path)
        desc += 'Load Path: {}\n'.format(self.load_path)
        desc += 'Store Path: {}\n'.format(self.store_path)
        desc += 'Log Path: {}\n'.format(self.log_path)
        desc += 'Batches: {}\n'.format(self.batches)
        desc += 'Sample Size: {}\n'.format(self.sample_size)
        desc += 'Sammple Seed: {}\n'.format(self.sample_seed)

        return desc
