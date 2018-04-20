import h5py
import numpy as np
import re       
import os
import random
        
from . import converter
        
class Storage(object):
    PATCH_SIZE = 'patch-size'
    SCALE = 'scale'
    INPUTS = 'inputs'
    PATCH_NUMBER = 'patch-number'
    GROUND_TRUTH = 'gt'
    ORIGINAL = 'org'
    FEEDBACK = 'feedback'
    INDICES = 'idxs' 
    FEEDBACK_IMAGES = 'feedback-images'
    WIDTH = 'width'
    HEIGHT = 'height'
    AUGMENTED = 'augmented'
    PATH_FMT_SHORT = '{}/{}'
    PATH_FMT = '{0}/{1}/{2}'
    TOP_LEFT = 'top-left'   
    BOTTOM_LEFT = 'bottom-left'
    WEIGHTS = 'weights'

    def __init__(self, storage_path, write):
        if write:
            self.storage = h5py.File(storage_path, 'a')
        else:
            self.storage = h5py.File(storage_path, 'r')
        
    def __enter__(self):                       
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    def close(self):
        self.storage.close()
        
    def _normalize_name(self, name):
        return name.split(os.path.sep)[-1].split('.')[0].lower()

    @staticmethod
    def org_path(dataset, name):
        return Storage.PATH_FMT.format(dataset, Storage.ORIGINAL, name)

    @staticmethod
    def gt_path(dataset, name=None):
        if name is None:
            path = Storage.PATH_FMT_SHORT.format(dataset, Storage.GROUND_TRUTH)
        else:
            path = Storage.PATH_FMT.format(dataset, Storage.GROUND_TRUTH, name)
        return path

    @staticmethod
    def weights_path(dataset, name):
        return Storage.PATH_FMT.format(dataset, Storage.WEIGHTS, name)

class StorageWriter(Storage):
    """
        This class is used to create and update HDFS5 databases.
    """
    def __init__(self, storage_path, converter=None):
        super(StorageWriter, self).__init__(storage_path, True)
        self.converter = converter
        if self.converter is not None:
            self.__init_database_attrs(converter)
        self.total_patch_number = 0
        self._img_pattern = re.compile('.*\.(tiff|tif|bmp|png|jpg)$')
        self._pweight_pattern = re.compile('(.*)_PWeights.dat$')
        self._rweight_pattern = re.compile('(.*)_RWeights.dat$')

    def _init_attr(self, key, val):
        if key in self.storage.attrs.keys():
            if val != self.storage.attrs[key]:
                raise ValueError('Value mismatch for {0}: {1} != {2}'
                                    .format(key, self.storage.attrs[key], val))
        else:
            self.storage.attrs.create(key, val)

    def __init_database_attrs(self, converter):
        self._init_attr(self.PATCH_SIZE, converter.patch_size)
        scale = converter.scale
        self._init_attr(self.SCALE, scale)
        self._init_attr(self.INPUTS, converter.inputs)

    def close(self):
        if self.PATCH_NUMBER in self.storage.attrs.keys():
            self.storage.attrs[self.PATCH_NUMBER] += self.total_patch_number
        else:
            self.storage.attrs.create(self.PATCH_NUMBER, self.total_patch_number)
        super(StorageWriter, self).close()

    def __add_gt_image(self, name, path, grp):
        conv, x, y = self.converter.convert(path, False)
        dset = grp.create_dataset(name, data=conv)
        dset.attrs.create(self.WIDTH, x)
        dset.attrs.create(self.HEIGHT, y)
        return conv.shape[0]

    def __add_org_image(self, name, path, grp):
        conv1, conv2, x, y = self.converter.convert(path, True)
        img_grp = grp.create_group(name)
        img_grp.attrs.create(self.WIDTH, x)
        img_grp.attrs.create(self.HEIGHT, y)
        img_grp.create_dataset(self.TOP_LEFT, data=conv1)
        img_grp.create_dataset(self.BOTTOM_LEFT, data=conv2)
        return conv1.shape[0]

    def _add_image(self, name, path, grp, org):
        if org:
            patch_number = self.__add_org_image(name, path, grp)
        else:
            patch_number = self.__add_gt_image(name, path, grp)
        return patch_number

    def _add_datasets(self, grp, names, base_path, org):
        patch_number = 0
        for name in names:
            path = os.path.join(base_path, name)
            name = name.split('.')[0].lower()
            patch_number += self._add_image(name, path, grp, org)
        return patch_number

    def _add_weights(self, grp, dataset, p_names, r_names, base_path):
        p_names.sort()
        r_names.sort()
        for p, r in zip(p_names, r_names):
            p_name = p.split('_')[0].lower()
            r_name = r.split('_')[0].lower()
            assert p_name == r_name, "adding weights: {} != {}".format(p_name,
                                                                        r_name)
            gt_img = self.storage[Storage.gt_path(dataset, p_name)]
            p_path = os.path.join(base_path, p)
            r_path = os.path.join(base_path, r)
            weights = self.converter.convert_weights(p_path, r_path,
                    gt_img.attrs[self.WIDTH], gt_img.attrs[self.HEIGHT])
            dset = grp.create_dataset(p_name, data=weights)

    def store_dataset(self, dataset, org_path, gt_path, wgt_path):
        org_names = [f for f in os.listdir(org_path)
                        if self._img_pattern.match(f) is not None]
        gt_names = [f for f in os.listdir(gt_path)
                        if self._img_pattern.match(f) is not None]
        pw_names = [f for f in os.listdir(wgt_path)
                        if self._pweight_pattern.match(f) is not None] 
        rw_names = [f for f in os.listdir(wgt_path)
                        if self._rweight_pattern.match(f) is not None] 

        grp = self.storage.create_group(dataset.lower())
        org_grp = grp.create_group(self.ORIGINAL)
        gt_grp = grp.create_group(self.GROUND_TRUTH)
        wgt_grp = grp.create_group(self.WEIGHTS)

        patch_number = self._add_datasets(org_grp, org_names, org_path, True)
        grp.attrs.create(self.PATCH_NUMBER, patch_number)
        self.total_patch_number += patch_number
        self._add_datasets(gt_grp, gt_names, gt_path, False)
        self._add_weights(wgt_grp, dataset, pw_names, rw_names, wgt_path)

    def __add_group(self, group, name):
        if name in group.keys():
            grp = group[name]
        else:
            grp = group.create_group(name)
        return grp

    def store_image(self, dataset, path):
        grp = self.__add_group(self.storage, dataset.lower())
        org_grp = self.__add_group(grp, self.ORIGINAL)
        # extract the actual file name from path
        name = os.path.splitext(os.path.basename(path))[0].lower()
        patch_number = self._add_image(name, path, org_grp, True)
        grp.attrs.create(self.PATCH_NUMBER, patch_number)
        self.total_patch_number += patch_number

class StorageReader(Storage):
    def __init__(self, storage_path, train=None, test=None, validation=None,
                        batch_size=None, with_weights=False):
        super(StorageReader, self).__init__(storage_path, False)
        self.train = train
        self.test = test
        self.validation = validation
        self.batch_size = batch_size
        self.with_weights = with_weights

    def patch_size(self):
        return self.storage.attrs[self.PATCH_SIZE]

    def inputs(self):
        return self.storage.attrs[self.INPUTS]

    def sequence_length(self):
        return self.patch_size()**2 / self.inputs()

    def dataset_images(self, dataset):
        path = '{0}/org/'.format(dataset)
        return self.storage[path].keys()

    def test_iterator(self):
        return StorageIterator(self.storage, self.test, self.batch_size,
                                    self.with_weights)

    def validation_iterator(self):
        return StorageIterator(self.storage, self.validation, self.batch_size,
                                    self.with_weights)

    def random_train_iterator(self, max_batches):
        return RandomStorageIterator(self.storage, self.train, self.batch_size,
                                        max_batches, self.with_weights)

    def random_test_iterator(self, max_batches):
        return RandomStorageIterator(self.storage, self.test, self.batch_size,
                                        max_batches, self.with_weights)

    def random_validation_iterator(self, max_batches):
        return RandomStorageIterator(self.storage, self.validation, self.batch_size,
                                        max_batches, self.with_weights)

    def read_img(self, dataset, image):
        path = self.PATH_FMT.format(dataset, self.ORIGINAL,
                                        self._normalize_name(image))
        dset = self.storage[path]
        x = dset.attrs[self.WIDTH]
        y = dset.attrs[self.HEIGHT]
        tl = dset[Storage.TOP_LEFT]
        bl = dset[Storage.BOTTOM_LEFT]
        return np.array([tl, bl]), x, y, tl.shape[0]

    def image_iterator(self, image_name, image_dataset):
        return ImageIterator(self.storage, self.batch_size, image_name,
                image_dataset)

class BasicStorageIterator(object):

    def __init__(self, storage, batch_size, with_weights=False):
        self._storage = storage
        self._batch_size = batch_size
        self._list_index = 0
        self._ds_list = None
        self._with_weights = with_weights

    def __iter__(self):
        return self

    def _next_batch_from(self, ds_list):
        gt = []
        org = []
        wgts = []
        new_idx = self._list_index + self._batch_size
        if new_idx > len(ds_list):
            raise StopIteration

        for i in range(self._list_index, new_idx):
            element = ds_list[i]
            o, g, w = self._fetch_elements(element)
            org.append(o)
            gt.append(g)
            wgts.append(w)

        self._list_index = new_idx
 	# change format from [batches, dims, input seq, inputs] to [dims, batches, input seq, inputs]
        tmp = np.array(org)
        return tmp.swapaxes(0,1), gt, wgts

    def _fetch_elements(self, element):
        tl = self._storage[element[0]][Storage.TOP_LEFT][element[3]]
        bl = self._storage[element[0]][Storage.BOTTOM_LEFT][element[3]]
        o = np.array([tl, bl])
        idx = len(element) - 1
        g = self._storage[element[1]][element[idx]]
        if self._with_weights:
            w = self._storage[element[2]][element[idx]]
        else:
            w = None
        return o, g, w
   
    def _find_batch_size(self, instances):
        batch_size = instances
        for i in reversed(range(1, 33)):
            if instances % i == 0:
                batch_size = i
                break
        return batch_size

    def _list_image_patches(self, ds, img):
        lst = []
        gt_path = Storage.gt_path(ds, img)
        org_path = Storage.org_path(ds, img)
        wgts_path = Storage.weights_path(ds, img)
        for i in range(self._storage[gt_path].shape[0]):
            lst.append((org_path, gt_path, wgts_path, i))
        return lst

    def _init_dataset_list(self, datasets):
        """Initialize list of tuples containing all patches of the configured
        datasets.
            
            Returns:
                list of tuples: [(original_hdfs_path, gt_hdfs_path, patch_id)]
        """
        lst = []
        for ds in datasets:
            if isinstance(ds, basestring):
                base_path = Storage.gt_path(ds)
                for img in self._storage[base_path].keys():
                    lst += self._list_image_patches(ds, img)
            else:
                lst += self._list_image_patches(ds[0],
                                                Storage.normalize_name(ds[1]))
        return lst 

    def reset(self):
        self._list_index = 0

    def batches(self):
        return len(self._ds_list) / self._batch_size

    def batch_size(self):
        return self._batch_size

    def items(self):
        return len(self._ds_list)

    def next(self):
        return self._next_batch_from(self._ds_list)

class ImageIterator(BasicStorageIterator):

    def __init__(self, storage, batch_size, img_name, img_dataset):
        if batch_size is None:
            path = Storage.PATH_FMT.format(img_dataset, Storage.GROUND_TRUTH, img_name)
            instances = storage[path].shape[0]
            batch_size = self._find_batch_size(instances)
        super(ImageIterator, self).__init__(storage, batch_size, False)
        self._ds_list = self._list_image_patches(img_name, img_dataset)

class StorageIterator(BasicStorageIterator):

    def __init__(self, storage, datasets, batch_size, with_weights=False):
        if batch_size is None:
            instances = self.__get_instances(storage, datasets)
            batch_size = self._find_batch_size(instances)
        super(StorageIterator, self).__init__(storage, batch_size, with_weights)
        self._datasets = datasets
        self._ds_list = self._init_dataset_list(self._datasets)

    def __get_instances(self, storage, datasets):
        instances = 0
        for ds in datasets:
            instances += storage[ds].attrs[Storage.PATCH_NUMBER]
        return instances

class RandomStorageIterator(StorageIterator):

    def __init__(self, storage, datasets, batch_size, max_batches,
                                                    with_weights=False):
        """
            Iterator to return ranomly selected patches from the given datasets.

            storage: storage to operate on
            datasets: datasets to sample from
            batch_size: how many patches should be returned per iteration step
            max_batches: maximium number of patches per epoch
        """
        super(RandomStorageIterator, self).__init__(storage, datasets,
                batch_size, with_weights)
        self._max_batches = max_batches
        self._sample = self.__random_sample()

    def __random_sample(self):
        if len(self._ds_list) > self._max_batches:
            sample = random.sample(self._ds_list, self._max_batches)
        else:
            sample = self._ds_list
        return sample

    def next(self):
        return self._next_batch_from(self._sample)

    def batches(self):
        return self._max_batches

    def reset(self):
        self._list_index = 0
        self._sample = self.__random_sample()
