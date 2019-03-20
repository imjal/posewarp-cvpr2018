import tensorflow as tf
import os
import numpy as np
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import truncated_vgg
from keras.optimizers import Adam
import cv2
import pdb


def train(model_name, gpu_id):
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
    config.allow_soft_placement = True

    gan_lr = 1e-4
    disc_lr = 1e-4
    disc_loss = 0.1

    generator = networks.network_posewarp(params)
    generator.load_weights('/home/jl5/posewarp-cvpr2018/models/orig-model-gan/9000.h5')

    discriminator = networks.discriminator(params)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=disc_lr))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')

    gan = networks.gan(generator, discriminator, params)
    gan.compile(optimizer=Adam(lr=gan_lr),
                loss=[networks.vgg_loss(vgg_model, response_weights, 12), 'binary_crossentropy'],
                loss_weights=[1.0, disc_loss])

    n_iters = 10000
    batch_size = params['batch_size']

    for step in range(n_iters):

        x, y = next(train_feed)

        arr_loss = generator.predict_on_batch(x)
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
        train(sys.argv[1], sys.argv[2])
