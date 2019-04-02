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
from html4vision import Col, imagetable


def test(model_name, gpu_id):
    params = param.get_general_params()

    network_dir = params['model_save_dir'] + '/' + model_name
    save_dir = params['save_img_dir'] + '/' + model_name

    if not os.path.isdir(network_dir):
        os.mkdir(network_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    train_feed = data_generation.create_feed_canon(params, params['data_dir'], 'train', do_augment = False)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)
    model.load_weights('/home/jl5/posewarp-cvpr2018/models/orig-train-set/18000.h5')
    #model.compile(optimizer=Adam(lr=1e-4), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])

    model.summary()
    n_iters = params['n_training_iter']

    for step in range(50):
        x, y = next(train_feed)
        arr_out = model.predict_on_batch(x)
        for i in range(params['batch_size']):
            for j in range(11):
                cv2.imwrite(save_dir + '/' +  str(step) + '_' + str(i) + "_limb" + str(j)+  ".png", ((x[3][i][..., j] + 1)*128))
                cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + '_warped' + str(j) + '.png', (arr_out[1][i][..., 3*j:3*j+3]+1)*128)

            cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'generated.png', ((arr_out[0][i] + 1) * 128).astype('uint8'))
            cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'source'+ '.png', ((x[0][i] + 1) * 128).astype('uint8'))
            cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'target' +'.png', ((y[i] + 1) * 128).astype('uint8'))
            #cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'bg_synth' +'.png', ((arr_out[1][i] + 1) * 128).astype('uint8'))
            #cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'fg_synth' +'.png', ((arr_out[2][i] + 1) * 128).astype('uint8'))
            #cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'bg_output' +'.png', ((y[i][2] + 1) * 128).astype('uint8'))
            #cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'fg_output' +'.png', ((y[i][3] + 1) * 128).astype('uint8'))
            #cv2.imwrite(save_dir + '/' + str(step) + '_' + str(i) + 'src_mask' +'.png', ((y[i][3] + 1) * 128).astype('uint8'))
    cols = [
        Col('id1', 'ID'), # make a column of 1-based indices
        Col('img', 'Source',  '../saved_imgs/' + model_name + '/*source.png'), # specify image content for column 2
        Col('img', 'Target',  '../saved_imgs/' + model_name + '/*target.png'), # specify image content for column 3
        Col('img', 'Generated', '../saved_imgs/' + model_name  + '/*generated.png'),
        #Col('img', 'Foreground Synth', '../saved_imgs/' + model_name  + '/*fg_synth.png'),
        #Col('img', 'Background Synth', '../saved_imgs/' + model_name  + '/*bg_synth.png'),
    ]

    cols1 = []
    for i in range(11):
        cols1 += [Col('img', 'limb ' + str(i),  '../saved_imgs/' + model_name + '/*limb' + str(i) + "*")]
    cols2 = []
    for i in range(11):
        cols2 += [Col('img', 'Limb' + str(i),  '../saved_imgs/' + model_name + '/*_warped' + str(i) + "*")]

    # html table generation
    imagetable(cols, outfile='../saved_results/' + model_name + '.html')
    imagetable(cols1, outfile='../saved_results/' + model_name + "limbs" + '.html')
    imagetable(cols2, outfile='../saved_results/' + model_name + "warped_limbs" + '.html')



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need model name and gpu id as command line arguments.")
    else:
        import pdb
        test(sys.argv[1], sys.argv[2])
