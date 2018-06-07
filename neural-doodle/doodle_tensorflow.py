import tensorflow as tf
import numpy as np
import skimage.io
import itertools
import os
import bz2
import argparse
import scipy
import skimage.io
import skimage.color

CONTENT_LAYERS = ['4_2']
STYLE_LAYERS = ['3_1', '4_1']

def conv2d(input_tensor, kernel, bias):
    # Theano的kernel格式是[ouput_channels, input_channels, width, height]
    # 而tf的格式是[width, height, input_channels, output_channels]
    kernel = np.transpose(kernel, [2, 3, 1, 0])
    # 卷积的参数是kernel=(3,3), stride=1, padding=0；在卷积之前，每个维度pad 2，可以保证输入输出大小一致
    x = tf.pad(input_tensor, [[0,0], [1,1], [1,1], [0,0]])
    x = tf.nn.conv2d(x, tf.constant(kernel), (1,1,1,1), 'VALID')
    x = tf.nn.bias_add(x, tf.constant(bias))
    return tf.nn.relu(x)

def avg_pooling(input_tensor, size=2):
    return tf.nn.pool(input_tensor, [size, size], 'AVG', 'VALID', strides=[size, size])

def norm(arr):
    n, *shape = arr.shape
    lst = []
    for i in range(n):
        v = arr[i, :].flatten()
        v /= np.sqrt(sum(v**2))
        lst.append(np.reshape(v, shape))
    return lst

def build_base_net(input_tensor, mask_depth=1):
    vgg19_file = os.path.join(os.path.dirname(__file__), 'vgg19_conv.pkl.bz2')
    assert os.path.exists(vgg19_file), ("Model file with pre-trained convolution layers not found. Download here: "
        +"https://github.com/alexjc/neural-doodle/releases/download/v0.0/vgg19_conv.pkl.bz2")

    data = np.load(bz2.open(vgg19_file, 'rb'))
    k = 0
    net = {}
    # 网络分两块，main和map，main是vgg，map对应着vgg把mask下采样，也有自己的输入和输出
    # Primary network for the main image. These are convolution only, and stop at layer 4_2 (rest unused).
    net['img']     = input_tensor
    net['conv1_1'] = conv2d(net['img'], data[k], data[k+1])
    k += 2
    net['conv1_2'] = conv2d(net['conv1_1'], data[k], data[k+1])
    k += 2
    # average pooling without padding，第二个参数是pool size
    net['pool1']   = avg_pooling(net['conv1_2'])
    net['conv2_1'] = conv2d(net['pool1'], data[k], data[k+1])
    k += 2
    net['conv2_2'] = conv2d(net['conv2_1'], data[k], data[k+1])
    k += 2
    net['pool2']   = avg_pooling(net['conv2_2'])
    net['conv3_1'] = conv2d(net['pool2'], data[k], data[k+1])
    k += 2
    net['conv3_2'] = conv2d(net['conv3_1'], data[k], data[k+1])
    k += 2
    net['conv3_3'] = conv2d(net['conv3_2'], data[k], data[k+1])
    k += 2
    net['conv3_4'] = conv2d(net['conv3_3'], data[k], data[k+1])
    k += 2
    net['pool3']   = avg_pooling(net['conv3_4'])
    net['conv4_1'] = conv2d(net['pool3'], data[k], data[k+1])
    k += 2
    net['conv4_2'] = conv2d(net['conv4_1'], data[k], data[k+1])
    k += 2
    net['conv4_3'] = conv2d(net['conv4_2'], data[k], data[k+1])
    k += 2
    net['conv4_4'] = conv2d(net['conv4_3'], data[k], data[k+1])
    k += 2
    net['pool4']   = avg_pooling(net['conv4_4'])
    net['conv5_1'] = conv2d(net['pool4'], data[k], data[k+1])
    k += 2
    net['conv5_2'] = conv2d(net['conv5_1'], data[k], data[k+1])
    k += 2
    net['conv5_3'] = conv2d(net['conv5_2'], data[k], data[k+1])
    k += 2
    net['conv5_4'] = conv2d(net['conv5_3'], data[k], data[k+1])
    k += 2
    net['main']    = net['conv5_4']

    # Auxiliary network for the semantic layers, and the nearest neighbors calculations.
    net['map'] = tf.placeholder(tf.float32, (None, None, None, mask_depth))
    for j, i in itertools.product(range(5), range(4)):
        if j < 2 and i > 1: continue #和上面的convj_i对应，因为conv1和conv2都只有两个卷积层
        suffix = '%i_%i' % (j+1, i+1)

        if i == 0: #对mask下采样，缩小1, 2, 4, 8, 16, 32倍
            net['map%i'%(j+1)] = avg_pooling(net['map'], 2**j)

        # 默认值是10
        net['sem'+suffix] = tf.concat([net['conv'+suffix], net['map%i'%(j+1)]], -1)
        # dup层是单独的输入，nn层是对dup卷积
        #shape = [i.value for i in net['sem'+suffix].get_shape()]
        # net['dup'+suffix] = tf.placeholder(tf.float32, shape)
        # # 计算的是correlation
        # net['nn'+suffix] = conv2d(net['dup'+suffix], 1, 3, b=None, pad=0, flip_filters=False)
    return net

def extract_target_data(content, content_mask, style, style_mask):
    pixel_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,1,1,3))
    input_tensor = tf.placeholder(np.float32, shape=(None,None,None,3))
    net = build_base_net(input_tensor, content_mask.shape[-1])
    style_features = [net['sem'+layer] for layer in STYLE_LAYERS]
    content_features = [net['conv'+layer] for layer in CONTENT_LAYERS]
    tensors = []
    for f in style_features:
        dim = f.get_shape()[-1].value
        # x = (batch, height, width, patches)
        x = tf.extract_image_patches(f, (1,3,3,1), (1,1,1,1), (1,1,1,1), 'VALID')
        # x = (-1, patch_heigth, patch_width, channles)
        tensors.append(tf.reshape(x, (-1, 3, 3, dim)))
    style_data = []
    content_data = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for c in content_features:
            t = c.eval(feed_dict={net['img']:content-pixel_mean,
                net['map']:content_mask})
            content_data.append(t)
        for s in tensors:
            t = s.eval(feed_dict={net['img']:style-pixel_mean,
                net['map']:style_mask})
            style_data.append(t)
    return content_data, style_data

def format_and_norm(arr, depth, sem_weight):
    """
    BHWC 转换成 HWCB，并针对每个patch进行norm
    """
    n, *shape = arr.shape
    norm = np.zeros(shape+[n], dtype=arr.dtype)
    un_norm = np.zeros(shape+[n], dtype=arr.dtype)
    for i in range(n):
        t = arr[i, ...]
        un_norm[..., i] = t
        # 卷积层和mask层分开norm
        t1 = t[..., :depth]
        t1 = t1/np.sqrt(3*np.sum(t1**2)+1e-6)
        t2 = t[..., depth:]
        t2 = t2/np.sqrt(sem_weight*np.sum(t2**2)+1e-6)
        norm[..., i] = np.concatenate([t1,t2], -1)
    return norm, un_norm

class Model(object):
    def __init__(self, args, content, style, content_mask=None, style_mask=None):
        self.args = args
        self.pixel_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,1,1,3))

        self.content = np.expand_dims(content, 0).astype(np.float32)
        self.style = np.expand_dims(style, 0).astype(np.float32)
        if content_mask is not None:
            self.content_mask = np.expand_dims(content_mask, 0).astype(np.float32)
        else:
            self.content_mask = np.ones(self.content.shape[:-1]+(1,), np.float32)
        if style_mask is not None:
            self.style_mask = np.expand_dims(style_mask, 0).astype(np.float32)
        else:
            self.style_mask = np.ones(self.style.shape[:-1]+(1,), np.float32)
        assert self.content_mask.shape[-1] == self.style_mask.shape[-1]
        self.mask_depth = self.content_mask.shape[-1]
        self.content_data, self.style_data = extract_target_data(self.content, self.content_mask, self.style, self.style_mask)
        tf.reset_default_graph()
        input_tensor = tf.Variable(self.content)
        self.net = build_base_net(input_tensor, self.mask_depth)
        # content只在conv层比较
        self.content_features = [self.net['conv'+layer] for layer in CONTENT_LAYERS]
        self.style_features = [self.net['sem'+layer] for layer in STYLE_LAYERS]
        self.style_loss = 0
        for i in range(len(STYLE_LAYERS)):
            sem = self.style_features[i]
            # x = (batch, height, width, patches)
            patches = tf.extract_image_patches(sem, (1,3,3,1), (1,1,1,1), (1,1,1,1), 'VALID')
            # x = (-1, patch_heigth, patch_width, channles)
            # patches = tf.stop_gradient(patches)
            patches = tf.reshape(patches, (-1, 3, 3, sem.shape[-1].value))
            pow2 = patches**2
            p1 = tf.reduce_sum(pow2[..., :-self.mask_depth], [1,2,3])
            p1 = tf.reshape(p1, [-1,1,1,1])
            p1 = pow2[..., :-self.mask_depth]/(3*p1+1e-6)
            p2 = tf.reduce_sum(pow2[..., -self.mask_depth:], [1,2,3])
            p2 = tf.reshape(p2, [-1,1,1,1])
            p2 = pow2[..., -self.mask_depth:]/(self.args.semantic_weight*p2+1e-6)
            norm_patch = tf.concat([p1, p2], -1)
            norm_patch = tf.reshape(norm_patch, [-1, 9*sem.shape[-1].value])
            norm, un_norm = format_and_norm(self.style_data[i], -self.mask_depth, self.args.semantic_weight)
            norm = np.reshape(norm, [9*sem.shape[-1].value, -1])
            # 这个分支不需要纳入梯度计算中
            # sim = tf.stop_gradient(tf.nn.conv2d(norm_patch, tf.constant(norm), (1,1,1,1), 'VALID'))
            # sim = tf.squeeze(sim)
            sim = tf.matmul(norm_patch, norm)
            self.net['sim'+STYLE_LAYERS[i]] = sim
            # n = sim.shape[-1].value
            # b, h, w, c = map(lambda x:x.value, sem.shape)
            max_ind = tf.argmax(sim, axis=-1)
            target_patches = tf.gather(self.style_data[i], tf.reshape(max_ind, [-1]))
            # max_one_hot = tf.one_hot(max_ind, n)
            # self.net['onehot'+STYLE_LAYERS[i]] = max_one_hot
            # match_patch = tf.nn.conv2d_transpose(max_one_hot, tf.constant(un_norm), 
            #     sem.shape, (1,1,1,1), padding='VALID')
            
            # # 转置卷积之后，patch之间有重合，取了平均值
            # _, c_height, c_width, _ = sem.shape
            # patch_size = 3
            # mask = tf.ones((c_height, c_width), tf.float32)
            # fullmask = tf.zeros((c_height+patch_size-1, c_width+patch_size-1), tf.float32)
            # for x in range(patch_size):
            #     for y in range(patch_size):
            #         paddings = [[x, patch_size-x-1], [y, patch_size-y-1]]
            #         padded_mask = tf.pad(mask, paddings=paddings, mode="CONSTANT")
            #         fullmask += padded_mask
            # pad_width = int((patch_size-1)/2)
            # overlaps = tf.slice(fullmask, [pad_width, pad_width], [c_height, c_width])
            # overlaps = tf.reshape(overlaps, shape=(1, c_height, c_width, 1))
            # match_patch /= overlaps
            # self.net['match'+STYLE_LAYERS[i]] = match_patch
            self.style_loss += tf.reduce_mean((patches[...,:-self.mask_depth]-target_patches[...,:-self.mask_depth])**2)
        self.style_loss *= args.style_weight
        self.content_loss = 0
        for c, t in zip(self.content_features, self.content_data) :
            self.content_loss += tf.reduce_mean((c-t)**2)
        self.content_loss *= args.content_weight
        self.variation_loss = args.smoothness*tf.reduce_mean((input_tensor[..., :-1,:-1]
            -input_tensor[..., 1:, :-1])**2+(input_tensor[..., :-1, :-1]-input_tensor[..., :-1, 1:])**2)
        self.loss = self.style_loss + self.content_loss + self.variation_loss
        self.grad = tf.gradients(self.loss, self.net['img'])
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./summary', tf.get_default_graph())
    def evaluate(self):
        sess = tf.Session()
        def func(img):
            self.iter += 1
            current_img = img.reshape(self.content.shape).astype(np.float32) - self.pixel_mean

            feed_dict = {self.net['img']:current_img, self.net['map']:self.content_mask}
            loss = 0
            grads = 0
            style_loss = 0
            content_loss = 0
            sess.run(tf.global_variables_initializer())
            loss, grads, style_loss, content_loss, variation_loss, summ= sess.run(
                [self.loss, self.grad, self.style_loss, self.content_loss, self.variation_loss, self.merged],
                feed_dict=feed_dict)
            self.summary_writer.add_summary(summ, self.iter)
            if self.iter % 10 == 0:
                out = current_img + self.pixel_mean
                out = np.squeeze(out)
                out = np.clip(out, 0, 255).astype('uint8')
                skimage.io.imsave('outputs/%d-%s'%(self.iter, self.args.output), out)

            print('Iter:%d, loss: %f, style loss: %f, content loss: %f variation loss: %f.'%
                (self.iter, loss, style_loss, content_loss, variation_loss))
            if np.isnan(grads).any():
                raise OverflowError("Optimization diverged; try using a different device or parameters.")

            # Return the data in the right format for L-BFGS.
            return loss, np.array(grads).flatten().astype(np.float64)
        return func

    def run(self):
        """The main entry point for the application, runs through multiple phases at increasing resolutions.
        """
        args = self.args
        Xn = self.content.copy()
        self.iter = 0
        # Optimization algorithm needs min and max bounds to prevent divergence.
        data_bounds = np.zeros((np.product(Xn.shape), 2), dtype=np.float64)
        data_bounds[:] = (0.0, 255.0)
        try:
            Xn, *_ = scipy.optimize.fmin_l_bfgs_b(
                            self.evaluate(),
                            Xn.flatten(),
                            bounds=data_bounds,
                            factr=0.0, pgtol=0.0,            # Disable automatic termination, set low threshold.
                            m=5,                             # Maximum correlations kept in memory by algorithm.
                            maxfun=args.iterations,        # Limit number of calls to evaluate().
                            iprint=-1)                       # Handle our own logging of information.
        except OverflowError:
            print("The optimization diverged and NaNs were encountered.",
                    "  - Try using a different `--device` or change the parameters.",
                    "  - Make sure libraries are updated to work around platform bugs.")
        except KeyboardInterrupt:
            print("User canceled.")
        except Exception as e:
            print(e)

        # args.seed = 'previous'
        # Xn = Xn.reshape(self.content.shape[1:])+self.pixel_mean[0]

        # output = np.clip(Xn, 0, 255).astype('uint8')
        # skimage.io.imsave(args.output, output)
        self.summary_writer.close()
    def adam_optimize(self):
        train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)
        img = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(100):
                _, loss, style_loss, content_loss, summ = sess.run(
                    [train_step, self.loss, self.style_loss, self.content_loss, self.merged],
                    feed_dict={self.net['map']:self.content_mask})
                self.summary_writer.add_summary(summ, i)
                print('Iter %d, Loss %f, Style loss %f, Content loss %f.'%(
                    i, loss, style_loss, content_loss
                ))
            img = self.net['img'].eval()
        img = np.squeeze(img, 0) + self.pixel_mean[0]
        output = np.clip(img, 0, 255).astype('uint8')
        skimage.io.imsave(self.args.output, output)
        self.summary_writer.close()
def prepare_mask(content_mask, style_mask, n):
    from sklearn.cluster import KMeans
    x1 = content_mask.reshape((-1, content_mask.shape[-1]))
    x2 = style_mask.reshape((-1, style_mask.shape[-1]))
    kmeans = KMeans(n_clusters=n, random_state=0).fit(x1)
    y1 = kmeans.labels_
    y2 = kmeans.predict(x2)
    y1 = y1.reshape(content_mask.shape[:-1])
    y2 = y2.reshape(style_mask.shape[:-1])
    diag = np.diag([1 for _ in range(n)])
    return diag[y1].astype(np.float32), diag[y2].astype(np.float32)

def main():
    # Configure all options first so we can custom load other libraries (Theano) based on device specified by user.
    parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument

    add_arg('--content',        default=None, type=str,         help='Content image path as optimization target.')
    add_arg('--content-mask', default=None, type=str, help='Content image mask.')
    add_arg('--content-weight', default=10.0, type=float,       help='Weight of content relative to style.')
    add_arg('--content-layers', default='4_2', type=str,        help='The layer with which to match content.')
    add_arg('--style',          default=None, type=str,         help='Style image path to extract patches.')
    add_arg('--style-mask', default=None, type=str, help='Style image mask.')
    add_arg('--style-weight',   default=25.0, type=float,       help='Weight of style relative to content.')
    add_arg('--style-layers',   default='3_1,4_1', type=str,    help='The layers to match style patches.')
    add_arg('--semantic-ext',   default='_sem.png', type=str,   help='File extension for the semantic maps.')
    add_arg('--semantic-weight', default=10.0, type=float,      help='Global weight of semantics vs. features.')
    add_arg('--output',         default='output.jpg', type=str, help='Output image path to save once done.')
    add_arg('--output-size',    default=None, type=str,         help='Size of the output image, e.g. 512x512.')
    add_arg('--phases',         default=3, type=int,            help='Number of image scales to process in phases.')
    add_arg('--slices',         default=2, type=int,            help='Split patches up into this number of batches.')
    add_arg('--cache',          default=0, type=int,            help='Whether to compute matches only once.')
    add_arg('--smoothness',     default=1E+0, type=float,       help='Weight of image smoothing scheme.')
    add_arg('--variety',        default=0.0, type=float,        help='Bias toward selecting diverse patches, e.g. 0.5.')
    add_arg('--seed',           default='noise', type=str,      help='Seed image path, "noise" or "content".')
    add_arg('--seed-range',     default='16:240', type=str,     help='Random colors chosen in range, e.g. 0:255.')
    add_arg('--iterations',     default=100, type=int,          help='Number of iterations to run each resolution.')
    add_arg('--device',         default='cpu', type=str,        help='Index of the GPU number to use, for theano.')
    add_arg('--print-every',    default=10, type=int,           help='How often to log statistics to stdout.')
    add_arg('--save-every',     default=10, type=int,           help='How frequently to save PNG into `frames`.')
    add_arg('--class-num', default=5, type=int, help='Count of mask classes.')
    
    args = parser.parse_args()
    style = skimage.io.imread(args.style)
    style_mask = args.style_mask
    if style_mask:
        style_mask = skimage.io.imread(args.style_mask)
    #style_mask = skimage.color.rgb2gray(style_mask)
    content = skimage.io.imread(args.content)
    content_mask = args.content_mask
    if content_mask:
        content_mask = skimage.io.imread(content_mask)
    #content_mask = skimage.color.rgb2gray(content_mask)
    content_mask, style_mask = prepare_mask(content_mask, style_mask, args.class_num)
    model = Model(args, content, style, content_mask, style_mask)
    model.run()
    #model.adam_optimize()

if __name__ == '__main__':
    main()