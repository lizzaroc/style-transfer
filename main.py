# Style-transfer algorithm to learn CNNs and lots of stuff
# Inspired from https://medium.com/mlreview/making-ai-art-with-style-transfer-using-keras-8bb5fa44b216

CONTENT_IMAGE_PATH = "data/lion.jpeg"
STYLE_IMAGE_PATH = "data/starry-night.jpg"
GENERATED_IMAGE_PATH = "results/out500.jpg"

from keras import backend as K
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import time

target_height = 512
target_width = 512
target_size = (target_height, target_width)

c_image_original = Image.open(CONTENT_IMAGE_PATH)
c_image_size_original = c_image_original.size

c_image = load_img(path=CONTENT_IMAGE_PATH, target_size=target_size)
c_im_arr = img_to_array(c_image)
c_im_arr = K.variable(preprocess_input(np.expand_dims(c_im_arr, axis=0)), dtype='float32')

s_image = load_img(path=STYLE_IMAGE_PATH, target_size=target_size)
s_im_arr = img_to_array(s_image)
s_im_arr = K.variable(preprocess_input(np.expand_dims(s_im_arr, axis=0)), dtype='float32')

gen_im0 = np.random.randint(256, size=(target_width, target_height, 3)).astype('float64')
gen_im0 = preprocess_input(np.expand_dims(gen_im0, axis=0))
g_im_placeholder = K.placeholder(shape=(1, target_width, target_height, 3))

def get_feature_reps(x, layer_names, model):
    """
    Get feature representations of input x for one or more layers in a given model.
    """
    feat_matrices = []
    for ln in layer_names:
        selected_layer = model.get_layer(ln)
        feat_raw = selected_layer.output
        feat_raw_shape = K.shape(feat_raw).eval(session=tf_session)
        N_l = feat_raw_shape[-1]
        M_l = feat_raw_shape[1]*feat_raw_shape[2]
        feat_matrix = K.reshape(feat_raw, (M_l, N_l))
        feat_matrix = K.transpose(feat_matrix)
        feat_matrices.append(feat_matrix)
    return feat_matrices

def get_content_loss(F, P):
    content_loss = 0.5 * K.sum(K.square(F - P))
    return content_loss

def get_gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G

def get_style_loss(ws, Gs, As):
    style_loss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_gram_matrix(G)
        A_gram = get_gram_matrix(A)
        style_loss += w * 0.25 * K.sum(K.square(G_gram - A_gram))/ (N_l **2  * M_l ** 2)
    return style_loss

def get_total_loss(g_im_placeholder, alpha=1.0, beta=10000.0):
    F = get_feature_reps(g_im_placeholder, layer_names=[c_layer_name], model=gen_model)[0]
    Gs = get_feature_reps(g_im_placeholder, layer_names=s_layer_names, model=gen_model)
    content_loss = get_content_loss(F, P)
    style_loss = get_style_loss(ws, Gs, As)
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss

def calculate_loss(g_im_arr):
    """
    Calculate total loss using K.function
    """
    if g_im_arr.shape != (1, target_width, target_width, 3):
        g_im_arr = g_im_arr.reshape((1, target_width, target_height, 3))
    loss_fcn = K.function([gen_model.input], [get_total_loss(gen_model.input)])
    return loss_fcn([g_im_arr])[0].astype('float64')

def get_grad(g_im_arr):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if g_im_arr.shape != (1, target_width, target_height, 3):
        g_im_arr = g_im_arr.reshape((1, target_width, target_height, 3))
    grad_fcn = K.function([gen_model.input], 
                          K.gradients(get_total_loss(gen_model.input), [gen_model.input]))
    grad = grad_fcn([g_im_arr])[0].flatten().astype('float64')
    return grad

def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (target_width, target_height, 3):
        x = x.reshape((target_width, target_height, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

def save_original_size(x, target_size=c_image_size_original):
    x_im = Image.fromarray(x)
    x_im = x_im.resize(target_size)
    x_im.save(GENERATED_IMAGE_PATH)
    return x_im

from keras.applications import VGG16
from scipy.optimize import fmin_l_bfgs_b

tf_session = K.get_session()
content_model = VGG16(include_top=False, weights='imagenet', input_tensor=c_im_arr)
style_model = VGG16(include_top=False, weights='imagenet', input_tensor=s_im_arr)
gen_model = VGG16(include_top=False, weights='imagenet', input_tensor=g_im_placeholder)
c_layer_name = 'block4_conv2'
s_layer_names = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                ]

P = get_feature_reps(x=c_im_arr, layer_names=[c_layer_name], model=content_model)[0]
As = get_feature_reps(x=s_im_arr, layer_names=s_layer_names, model=style_model)
ws = np.ones(len(s_layer_names))/float(len(s_layer_names))

start = time.time()

iterations = 500
x_val = gen_im0.flatten()
xopt, f_val, info= fmin_l_bfgs_b(calculate_loss, x_val, fprime=get_grad,
                            maxiter=iterations, disp=True)

x_out = postprocess_array(xopt)
x_im = save_original_size(x_out)
print('Image saved')
end = time.time()
print('Time taken: {}'.format(end-start))