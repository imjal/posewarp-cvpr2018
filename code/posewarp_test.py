import tensorflow as tf
import os
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import cv2
import truncated_vgg
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from PIL import Image
import numpy as np


def test(model_name, gpu_id):
    params = param.get_general_params()

    network_dir = params['model_save_dir'] + '/' + model_name
    save_dir = params['save_img_dir'] + '/' + model_name

    if not os.path.isdir(network_dir):
        os.mkdir(network_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    train_feed = data_generation.create_feed_canon(params, params['data_dir'], 'train')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)
    model.load_weights('/home/jl5/posewarp-cvpr2018/models/orig-model/5000.h5')
    model.compile(optimizer=Adam(lr=1e-4), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])

    model.summary()
    n_iters = params['n_training_iter']

    for step in range(10):
        x, y = next(train_feed)
        arr_loss = model.predict_on_batch(x)
        for i in range(params['batch_size']): 
            img = arr_loss[i]
            cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + '.png', ((img + 1) * 128).astype('uint8'))
            cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'source'+ '.png', ((x[0][i] + 1) * 128).astype('uint8'))
            cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'target' +'.png', ((y[i] + 1) * 128).astype('uint8'))
            pdb.set_trace()



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need model name and gpu id as command line arguments.")
    else:
        import pdb
        test(sys.argv[1], sys.argv[2])
