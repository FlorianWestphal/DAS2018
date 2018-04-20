import numpy as np
from PIL import Image


class Converter(object):
    def __init__(self, patch_size, inputs):
        if patch_size is not None:
            self.patch_size = patch_size
        else:
            self.patch_size = 512
        self.inputs = inputs

    def patches(self, shape):
        x, y = shape
        x /= self.patch_size
        y /= self.patch_size
        return (x, y)

    def round_up(self, x, y):
        x, rx = divmod(x, self.patch_size)
        y, ry = divmod(y, self.patch_size)
        # add one to the respective dimension, if there is a rest
        x = (x + (rx != 0)) * self.patch_size
        y = (y + (ry != 0)) * self.patch_size

        return x, y


class ToSeqConverter(Converter):

    def __init__(self, config):
        super(ToSeqConverter, self).__init__(config.patch_size, config.inputs)

        self.preprocess = config.preprocess
        self.equalize = config.equalize
        self.scale = config.scale

    def __pad(self, f):
        """Pad the given image matrix to be divisible by self.patch_size."""

        x, y = f.shape
        x, y = self.round_up(x, y)
        # create zero initialized patch of patch_size * patch_size
        res = np.zeros((x, y))
        # insert the actual values into the larger patch
        res[:f.shape[0], :f.shape[1]] = f

        return res

    def __cut_blocks(self, patches, patch_number, inner):
        # subdivide patches into input blocks
        input_blocks = patches.reshape(patch_number, self.patch_size/inner,
                                        inner, -1, inner).swapaxes(2,3)
        # reshape matrix to form [patch_number, input_block_number, inputs]
        return input_blocks.reshape(patch_number, (self.patch_size/inner)**2,
                                    self.inputs)

    def __cut(self, f, patch_number, height, inner):
        # divide given matrix into patches
        patches = f.reshape(height/self.patch_size, self.patch_size, -1,
                            self.patch_size).swapaxes(1,2)
        return self.__cut_blocks(patches, patch_number, inner)
    
    def __normalize(self, f):
        lmin = float(f.min())
        lmax = float(f.max())

        return np.floor((f-lmin)/(lmax-lmin)*255.)

    def __equalize(self, f):
        h  = np.histogram(f, bins=256)[0]
        H = np.cumsum(h) / float(np.sum(h))
        e = np.floor(H[f.flatten().astype('int')]*255.)

        return e.reshape(f.shape)

    def __cut_context(self, context, patch_x, patch_y, inner):
        patch_number = patch_x * patch_y
        patches = np.empty([patch_number, self.patch_size, self.patch_size])
        step = self.patch_size / self.scale
        for i in range(patch_number):
            x = (i % patch_y) * step
            y = (i / patch_y) * step
            patches[i] = context[y:y+self.patch_size, x:x+self.patch_size]
        return self.__cut_blocks(patches, patch_number, inner)

    def __1d_cut(self, f, context):
        x, y = self.patches(f.shape)
        h, w = f.shape
        inner = int(np.sqrt(self.inputs))
        # reshape image matrix to form [patches, input sequences, inputs]
        r1 = self.__cut(f, x*y, h, inner)
        # offset image to allow reading the content of the previous input line
        prev = np.zeros((h,w))
        prev[inner:, :] = f[:-inner, :]
        r2 = self.__cut(prev, x*y, h, inner)
        # process context
        r3 = self.__cut_context(context, x, y, inner)
        return np.concatenate((r1, r2, r3), 2)

    def __1d_gt_cut(self, f):
        x, y = self.patches(f.shape)
        h, w = f.shape
        inner = int(np.sqrt(self.inputs))
        cut =  self.__cut(f, x*y, h, inner)
        return cut

    def __backflip(self, cut, width):
        patches, seq_len, inputs = cut.shape
        tmp = cut.reshape(-1, width/self.patch_size, seq_len, inputs)
        return np.flipud(tmp).reshape(patches, seq_len, inputs)

    def __2d_cut(self, f, context):
        # convert image matrix to sequence from top left to bottom right
        cut1 = self.__1d_cut(f, context)
        # convert image matrix to sequence from bottom left to top right
        tmp = np.flipud(f)
        tmp2 = np.flipud(context)
        cut2 = self.__1d_cut(tmp, tmp2)
        # align patches from cut1 and cut2
        cut2 = self.__backflip(cut2, f.shape[1])
        return cut1, cut2

    def __scale(self, img):
        x,y = img.size
        x /= self.scale
        y /= self.scale

        return img.resize((x, y), Image.ANTIALIAS)

    def __prepare_img(self, img_path, org):
        im = Image.open(img_path).convert('L')

	x, y = im.size
        imarray = np.array(im)

        if self.preprocess and org:
            imarray = self.__normalize(imarray)
            if self.equalize:
                imarray = self.__equalize(imarray)
        elif not org:
            # map groundtruth to values between 0 and 1
            imarray = np.divide(imarray, 255.0)

        im.close()
        return self.__pad(imarray), x, y

    def __pad_img(self, img):
        x, y = img.size
        margin = self.patch_size * self.scale - self.patch_size
        padded = Image.new('L', (x + margin,y + margin))
        corner = margin / 2
        padded.paste(img, (corner,corner))
        return padded

    def __prepare_context(self, padded):
        im = Image.fromarray(padded)
        padded = self.__pad_img(im)
        scaled = self.__scale(padded)
        return np.array(scaled) 

    def convert(self, img_path, org=True):
        """
        Convert given image to sequence

        args:
        img_path - path to the image to convert
        org - indicate if a original image or a ground truth image is converted
        
        returns:
        sequence - the converted sequence
        x - the original image width
        y - the original image height
        """
        
        padded, x, y = self.__prepare_img(img_path, org)
        if org:
            context = self.__prepare_context(padded)
            conv1, conv2 = self.__2d_cut(padded, context)
            return conv1, conv2, x, y
        else:
            return self.__1d_gt_cut(padded), x, y

    def __prepare_weights(self, p_path, r_path, x, y):
        p = np.fromfile(p_path, sep=' ')
        r = np.fromfile(r_path, sep=' ')
        p += 1.0
        weights = p + r
        return self.__pad(weights.reshape(x, y))
    
    def convert_weights(self, p_path, r_path, x, y):
        """
        Read precision and recall weights from their respective files and
        combine them into the weights used for the dynamically weighted cost
        function

        args:
        p_path  - path to the precision weights
        r_path  - path to the recall weights
        x       - image width
        y       - image height

        returns:
        sequence - the converted sequence with shape [bs, seq_len, inputs]
        """
        padded = self.__prepare_weights(p_path, r_path, x, y)
        return self.__1d_gt_cut(padded)

class FromSeqConverter(Converter):
    def __init__(self, patch_size):
        super(FromSeqConverter, self).__init__(patch_size, None)

    def convert(self, patches, x, y, path):
        r_x, r_y = self.round_up(x, y)
        p_x, p_y = self.patches((r_x, r_y))
        assert p_x * p_y == patches.shape[0], ("Given array has wrong shape: {} expected: {}".
                                               format(patches.shape, p_x*p_y))

        inner = int(np.sqrt(patches.shape[2]))
        tmp = patches.reshape(p_x * p_y, self.patch_size/inner, -1, inner,
                inner).swapaxes(2,3)
        result = tmp.reshape(r_y/self.patch_size, -1, self.patch_size, 
                self.patch_size).swapaxes(1,2).reshape(r_y, r_x)

        result = result.clip(0,1)
        im = Image.fromarray(np.uint8(result[:y, :x] * 255))
        im.save(path)










