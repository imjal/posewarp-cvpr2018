import tensorflow as tf
import os
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import truncated_vgg
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam


def train(model_name, gpu_id):
    params = param.get_general_params()

    network_dir = params['model_save_dir'] + '/' + model_name

    if not os.path.isdir(network_dir):
        os.mkdir(network_dir)

    train_feed = data_generation.create_feed_canon(params, params['data_dir'], 'train')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)
    #model.load_weights('/home/jl5/posewarp-cvpr2018/models/pretrained-vid0/23000.h5')
    model.compile(optimizer=Adam(lr=1e-4), loss=[networks.vgg_loss(vgg_model, response_weights, 12), networks.vgg_loss(vgg_model, response_weights, 12)], loss_weights = [1.0, 50.0])

    model.summary()
    n_iters = params['n_training_iter']

    for step in range(n_iters+1):
        x, y = next(train_feed)

        train_loss = model.train_on_batch(x, y)

        util.printProgress(step, 0, train_loss)

        if step > 0 and step % params['model_save_interval'] == 0:
            model.save(network_dir + '/' + str(step) + '.h5')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need model name and gpu id as command line arguments.")
    else:
        import pdb
        train(sys.argv[1], sys.argv[2])
