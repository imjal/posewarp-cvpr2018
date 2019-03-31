import os
import numpy as np
import cv2
import transformations
import scipy.io as sio
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import pdb
import pickle
import glob
from skimage.morphology import convex_hull_image


def make_vid_info_list(data_dir):
    vids = glob.glob(data_dir + '/frames/*')
    n_vids = len(vids)
    print(n_vids)
    vid_info = []

    for i in range(n_vids):
        path, vid_name = os.path.split(vids[i])
        info_name = data_dir + '/info/' + vid_name + '.mat'
        info = sio.loadmat(info_name)

        box = info['data']['bbox'][0][0]
        x = info['data']['X'][0][0]
        pdb.set_trace()

        vid_info.append([info, box, x, vids[i]])

    #show_box_and_keypoints('/home/jl5/posewarp-cvpr2018/data/train/frames/Flavia Pennetta in 4k/1.jpg', box, x)

    return vid_info

def show_box_and_keypoints(img_str, box, x, frame_num):

    im = np.array(Image.open(img_str), dtype=np.uint8)
    fig,ax = plt.subplots(1)
    x = x[:, :, frame_num] - 1.0
    plt.scatter(x[:, 0], x[:, 1])
    ax.imshow(im)
    coord = box[frame_num, :]
    rect = patches.Rectangle((coord[0],coord[1]),coord[2], coord[3],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()

def show_keypoints(im, x):
    fig,ax = plt.subplots(1)
    plt.scatter(x[:, 0], x[:, 1])
    ax.imshow(im)
    plt.show()
    plt.savefig("/home/jl5/tgt_pose.png")

def make_vid_info_list_pickled(data_dir):
    vids = glob.glob(data_dir + '/orig-frames/*')
    n_vids = len(vids)
    vid_path = "/home/jl5/pytorch-EverybodyDanceNow/code/"
    vid_info = []

    for i in range(9):
        keypoints = pickle.load(open(vid_path + "keypoints_vid"+str(i)+".pkl", "rb"))
        keypoints[:, 0, :] = keypoints[:, 0, :]*1.40625 + 280
        keypoints[:, 1, :] = keypoints[:, 1, :]*1.40625
        boxes = pickle.load(open(vid_path + "boxes_vid" + str(i) + ".pkl", "rb"))
        boxes[:, 0] = boxes[:, 0]*1.40625 + 280
        boxes[:, 1] = boxes[:, 1]*1.40625
        boxes[:, 2] = boxes[:, 2]*1.40625
        boxes[:, 3] = boxes[:, 3]*1.40625
        vid_info.append([{"X": keypoints, "bbox": boxes}, boxes, keypoints, "/data/jl5/data-posewarp/train/orig-frames/" + str(i)])
        #show_box_and_keypoints('/data/jl5/data-posewarp/train/orig-frames/' + str(i) + '/000001.png', boxes, keypoints, 0)
    return vid_info

def make_vid_info_list_pickled_vidnum(data_dir, i):
    vids = glob.glob(data_dir + '/orig-frames/*')
    n_vids = len(vids)
    vid_path = "/home/jl5/pytorch-EverybodyDanceNow/code/"
    vid_info = []

    keypoints = pickle.load(open(vid_path + "keypoints_vid"+str(i)+".pkl", "rb"))
    keypoints[:, 0, :] = keypoints[:, 0, :]*1.40625 + 280
    keypoints[:, 1, :] = keypoints[:, 1, :]*1.40625
    boxes = pickle.load(open(vid_path + "boxes_vid" + str(i) + ".pkl", "rb"))
    boxes[:, 0] = boxes[:, 0]*1.40625 + 280
    boxes[:, 1] = boxes[:, 1]*1.40625
    boxes[:, 2] = boxes[:, 2]*1.40625
    boxes[:, 3] = boxes[:, 3]*1.40625
    vid_info.append([{"X": keypoints, "bbox": boxes}, boxes, keypoints, "/data/jl5/data-posewarp/train/orig-frames/" + str(i)])
    #show_box_and_keypoints('/data/jl5/data-posewarp/train/orig-frames/' + str(i) + '/000001.png', boxes, keypoints, 0)
    return vid_info


def get_person_scale(joints):
    upper_body_size = (-joints[0][1] + (joints[8][1] + joints[11][1]) / 2.0)
    rcalf_size = np.sqrt((joints[9][1] - joints[10][1]) ** 2 + (joints[9][0] - joints[10][0]) ** 2)
    lcalf_size = np.sqrt((joints[12][1] - joints[13][1]) ** 2 + (joints[12][0] - joints[13][0]) ** 2)
    calf_size = (lcalf_size + rcalf_size) / 2.0

    rforearm_size = np.sqrt((joints[3][1] - joints[4][1]) ** 2 + (joints[3][0] - joints[4][0]) ** 2)
    lforearm_size = np.sqrt((joints[6][1] - joints[7][1]) ** 2 + (joints[6][0] - joints[7][0]) ** 2)
    forearm_size = (lforearm_size + rforearm_size) / 2.0
    size = np.max([2.5 * upper_body_size, 5.0 * forearm_size])
    if size <= 0:
        return 1
    return size / 200.0

"""
    img_name = os.path.join(vid_name, f'{frame_num +1:06d}' + '.png')
    if not os.path.isfile(img_name):
        img_name = os.path.join(vid_name, f'{frame_num +1:06d}' + '.jpg')
"""
def read_frame(vid_name, frame_num, box, x):

    img_name = os.path.join(vid_name, f'{frame_num +1:06d}' + '.png')
    if not os.path.isfile(img_name):
        img_name = os.path.join(vid_name, f'{frame_num +1:06d}' + '.jpg')
    """
    img_name = os.path.join(vid_name, str(frame_num + 1) + '.png')
    if not os.path.isfile(img_name):
        img_name = os.path.join(vid_name, str(frame_num + 1) + '.jpg')
    """
    img = cv2.imread(img_name)
    joints = x[:, :, frame_num] + 1.0
    box_frame = box[frame_num, :]
    scale = get_person_scale(joints)
    pos = np.zeros(2)
    pos[0] = (box_frame[0] + box_frame[2] / 2.0)
    pos[1] = (box_frame[1] + box_frame[3] / 2.0)

    return img, joints, scale, pos


def mask_torso(src_mask_prior, trans_in, i, img_width, img_height):
    T = np.repeat(np.expand_dims(src_mask_prior[i][..., 10], 3), 3, 2)
    T1 = 255 - T.astype('uint8')
    T1[T1 < 100] = 0
    T1[T1 >= 100] = 1
    warped_mask = cv2.warpAffine(T1, trans_in[i][..., 10], (img_width, img_height))
    return warped_mask

"""img = Image.new('L', (img_width, img_height), 0)
                ImageDraw.Draw(img).polygon(vertices, outline=1, fill=1)
                mask = np.array(img)
                mask_convex = convex_hull_image(mask)"""

def mask_torso_tgt(joints_tgt, i, img_width, img_height):
    torso = [2, 5, 8, 11]
    vertices = []
    for i in torso:
        if any(pt < 0 for pt in joints_tgt[i]):
            if i == 2:
                vertices.append([50, 0])
            if i == 5: 
                vertices.append([200, 0])
            if i == 8: 
                vertices.append([int(joints_tgt[2][0]), 250])
            if i == 11: 
                vertices.append([int(joints_tgt[5][0]), 250])
        vertices += [tuple([int(x) for x in joints_tgt[i]])]
    print(vertices)
    clusterMask = np.zeros((img_width, img_height))
    height = max(abs(joints_tgt[8][1] - joints_tgt[2][1]), abs(joints_tgt[11][1] - joints_tgt[5][1]))
    width = max(abs(joints_tgt[5][0] - joints_tgt[2][0]), abs(joints_tgt[11][0] - joints_tgt[8][0]))
    point = [min(vertices[0][0], vertices[1][0]), max(vertices[0][1], vertices[1][1])]
    clusterMask[point[1] : point[1] + int(height), point[0] : point[0] + int(width)] = 1
    print(height)
    print(width)
    print(point)
    pdb.set_trace()
    T = np.repeat(np.expand_dims(clusterMask, 2), 3, 2)
    return T


def warp_example_generator(vid_info_list, param, do_augment=True, return_pose_vectors=False):
    img_width = param['IMG_WIDTH']
    img_height = param['IMG_HEIGHT']
    pose_dn = param['posemap_downsample']
    sigma_joint = param['sigma_joint']
    n_joints = param['n_joints']
    scale_factor = param['obj_scale_factor']
    batch_size = param['batch_size']
    limbs = param['limbs']
    n_limbs = param['n_limbs']

    while True:
        x_src = np.zeros((batch_size, img_height, img_width, 3))
        x_mask_src = np.zeros((batch_size, img_height, img_width, n_limbs + 1))
        x_pose_src = np.zeros((batch_size, int(img_height / pose_dn), int(img_width / pose_dn), n_joints))
        x_pose_tgt = np.zeros((batch_size, int(img_height / pose_dn), int(img_width / pose_dn), n_joints))
        x_trans = np.zeros((batch_size, 2, 3, n_limbs + 1))
        x_posevec_src = np.zeros((batch_size, n_joints * 2))
        x_posevec_tgt = np.zeros((batch_size, n_joints * 2))
        x_torso_mask =  np.zeros((batch_size, img_height, img_width, 3))
        y = np.zeros((batch_size, img_height, img_width, 3))
        output_masked = np.zeros((batch_size, img_height, img_width, 3))

        i = 0
        while i < batch_size:

            # 1. choose random video.
            vid = np.random.choice(len(vid_info_list), 1)[0]

            vid_bbox = vid_info_list[vid][1]
            vid_x = vid_info_list[vid][2]
            vid_path = vid_info_list[vid][3]

            # 2. choose pair of frames
            n_frames = vid_x.shape[2]
            frames = np.random.choice(n_frames, 2, replace=False)
            while abs(frames[0] - frames[1]) / (n_frames * 1.0) <= 0.02:
                frames = np.random.choice(n_frames, 2, replace=False)

            I0, joints0, scale0, pos0 = read_frame(vid_path, frames[0], vid_bbox, vid_x)
            I1, joints1, scale1, pos1 = read_frame(vid_path, frames[1], vid_bbox, vid_x)



            if I0 is None:
                print("Image is None \n")
                continue

            if I1 is None:
                print("IMG2 is None\n")
                continue

            if scale0 > scale1:
                scale = scale_factor / scale0
            else:
                scale = scale_factor / scale1

            #if scale == 0:
                #pdb.set_trace()
            pos = (pos0 + pos1) / 2.0

            I0, joints0 = center_and_scale_image(I0, img_width, img_height, pos, scale, joints0)
            I1, joints1 = center_and_scale_image(I1, img_width, img_height, pos, scale, joints1)

            show_keypoints(I1, joints1)

            I0 = (I0 / 255.0 - 0.5) * 2.0
            I1 = (I1 / 255.0 - 0.5) * 2.0

            if do_augment:
                rflip, rscale, rshift, rdegree, rsat = rand_augmentations(param)
                I0, joints0 = augment(I0, joints0, rflip, rscale, rshift, rdegree, rsat, img_height, img_width)
                I1, joints1 = augment(I1, joints1, rflip, rscale, rshift, rdegree, rsat, img_height, img_width)



            posemap0 = make_joint_heatmaps(img_height, img_width, joints0, sigma_joint, pose_dn)
            posemap1 = make_joint_heatmaps(img_height, img_width, joints1, sigma_joint, pose_dn)

            src_limb_masks = make_limb_masks(limbs, joints0, img_width, img_height)
            src_bg_mask = np.expand_dims(1.0 - np.amax(src_limb_masks, axis=2), 2)
            src_masks = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)

            x_src[i, :, :, :] = I0
            x_pose_src[i, :, :, :] = posemap0
            x_pose_tgt[i, :, :, :] = posemap1
            x_mask_src[i, :, :, :] = src_masks
            x_trans[i, :, :, 0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            x_trans[i, :, :, 1:] = get_limb_transforms(limbs, joints0, joints1)

            x_posevec_src[i, :] = joints0.flatten()
            x_posevec_tgt[i, :] = joints1.flatten()

            y[i, :, :, :] = I1
            x_torso_mask[i, :, :, :] = mask_torso_tgt(joints1, i, img_width, img_height)
            output_masked[i, :, :, :] = y[i] * x_torso_mask[i]
            """cv2.imwrite("/home/jl5/source.png", (x_src[i]+1)*128)
            cv2.imwrite("/home/jl5/tgt_mask.png", (x_torso_mask[i]*255))
            cv2.imwrite("/home/jl5/target.png", (y[i]+1)*128)
            pdb.set_trace()"""

            i+=1
        out = [x_src, x_pose_src, x_pose_tgt, x_mask_src, x_trans, x_torso_mask]
        if return_pose_vectors:
            out.append(x_posevec_src)
            out.append(x_posevec_tgt)

        yield (out, [y, output_masked])


def create_feed(params, data_dir, mode, do_augment=True, return_pose_vectors=False, transfer=False):
    vid_info_list = make_vid_info_list(data_dir + '/' + mode)

    if transfer:
        feed = transfer_example_generator(ex_list, ex_list, params)
    else:
        feed = warp_example_generator(vid_info_list, params, do_augment, return_pose_vectors)

    return feed


def create_feed_canon(params, data_dir, mode, do_augment=True, return_pose_vectors=False, transfer=False):
    vid_info_list = make_vid_info_list_pickled(data_dir + '/' + mode)

    if transfer:
        feed = transfer_example_generator(ex_list, ex_list, params)
    else:
        feed = warp_example_generator(vid_info_list, params, do_augment, return_pose_vectors)

    return feed

def create_test_feed(params, data_dir, mode, do_augment=True, return_pose_vectors=False, transfer=False):
    vid_info_list = make_vid_info_list_pickled_vidnum(data_dir + '/' + mode, 9)

    if transfer:
        feed = transfer_example_generator(ex_list, ex_list, params)
    else:
        feed = warp_example_generator(vid_info_list, params, do_augment, return_pose_vectors)

    return feed

def create_vidi_feed(params, data_dir, mode, i, do_augment=True, return_pose_vectors=False, transfer=False):
    vid_info_list = make_vid_info_list_pickled_vidnum(data_dir + '/' + mode, i)

    if transfer:
        feed = transfer_example_generator(ex_list, ex_list, params)
    else:
        feed = warp_example_generator(vid_info_list, params, do_augment, return_pose_vectors)

    return feed

'''
def transfer_example_generator(examples0, examples1, param):
    img_width = param['IMG_WIDTH']
    img_height = param['IMG_HEIGHT']
    pose_dn = param['posemap_downsample']
    sigma_joint = param['sigma_joint']
    n_joints = param['n_joints']
    scale_factor = param['obj_scale_factor']
    batch_size = param['batch_size']
    limbs = param['limbs']

    while True:

        for i in xrange(batch_size):
            X_src = np.zeros((batch_size, img_height, img_width, 3))
            X_mask_src = np.zeros((batch_size, img_height, img_width, len(limbs) + 1))
            X_pose_src = np.zeros((batch_size, img_height / pose_dn, img_width / pose_dn, n_joints))
            X_pose_tgt = np.zeros((batch_size, img_height / pose_dn, img_width / pose_dn, n_joints))
            X_trans = np.zeros((batch_size, 2, 3, 11))
            Y = np.zeros((batch_size, img_height, img_width, 3))

            for i in xrange(batch_size):
                example0 = examples0[np.random.randint(0, len(examples0))]
                example1 = examples1[np.random.randint(0, len(examples1))]
                while (example0[-1] == example1[-1]):
                    example1 = examples1[np.random.randint(0, len(examples1))]

                I0, joints0, scale0, pos0 = read_example_info(example0[:32])
                I1, joints1, scale1, pos1 = read_example_info(example1[32:64])

                scale0 = scale_factor / example0[31]
                scale1 = scale_factor / example1[31]

                pos0 = np.array(example0[29:31])
                I0, joints0 = center_and_scale_image(I0, img_width, img_height, pos0, scale0, joints0)

                I1 = cv2.resize(I1, (0, 0), fx=scale1, fy=scale1)
                joints1 = joints1 * scale1
                offset = (joints0[10, :] + joints0[13, :] - joints1[10, :] - joints1[13, :]) / 2.0
                joints1 += np.tile(offset, (14, 1))
                T = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
                I1 = cv2.warpAffine(I1, T, (img_width, img_height))

                I0 = (I0 / 255.0 - 0.5) * 2.0
                I1 = (I1 / 255.0 - 0.5) * 2.0

                posemap0 = make_joint_heatmaps(img_height, img_width, joints0, sigma_joint, pose_dn)
                posemap1 = make_joint_heatmaps(img_height, img_width, joints1, sigma_joint, pose_dn)

                src_limb_masks = make_limb_masks(joints0, img_width, img_height)
                src_bg_mask = np.expand_dims(1.0 - np.amax(src_limb_masks, axis=2), 2)
                src_masks = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)

                X_src[i, :, :, :] = I0
                X_pose_src[i, :, :, :] = posemap0
                X_pose_tgt[i, :, :, :] = posemap1
                X_mask_src[i, :, :, :] = src_masks
                X_trans[i, :, :, 0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                X_trans[i, :, :, 1:] = get_limb_transforms(joints0, joints1)
                Y[i, :, :, :] = I1

            yield ([X_src, X_pose_src, X_pose_tgt, X_mask_src, X_trans], Y)


def actionExampleGenerator(examples,param,return_pose_vectors=False):

	img_width = param['IMG_WIDTH']
	img_height = param['IMG_HEIGHT']
	pose_dn = param['posemap_downsample']
	sigma_joint = param['sigma_joint']
	n_joints = param['n_joints']
	scale_factor = param['obj_scale_factor']

	while True:

		I0,joints0,scale0,pos0 = readExampleInfo(examples[0])
		scale = scale_factor/scale0
		I0,joints0 = centerAndScaleImage(I0,img_width,img_height,pos0,scale,joints0)
		posemap0 = makeJointHeatmaps(img_height,img_width,joints0,sigma_joint,pose_dn)

		src_limb_masks = makeLimbMasks(joints0,img_width,img_height)
		bg_mask = np.expand_dims(1.0 - np.amax(src_limb_masks,axis=2),2)
		src_masks = np.log(np.concatenate((bg_mask,src_limb_masks),axis=2)+1e-10)

		for i in range(1,len(examples)):
			I1,joints1,scale1,pos1 = readExampleInfo(examples[i])
			I1,joints1 = centerAndScaleImage(I1,img_width,img_height,pos0,scale,joints1)
			posemap1 = makeJointHeatmaps(img_height,img_width,joints1,sigma_joint,pose_dn)

			X_src = np.expand_dims(I0,0)
			X_pose_src = np.expand_dims(posemap0,0)
			X_pose_tgt = np.expand_dims(posemap1,0)
			X_mask_src = np.expand_dims(src_masks,0)
			X_trans = np.zeros((1,2,3,11))
			X_trans[i,:,:,0] = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
			X_trans[i,:,:,1:] = getLimbTransforms(joints0,joints1)
			X_posevec_src = np.expand_dims(joints0.flatten(),0)
			X_posevec_tgt = np.expand_dims(joints1.flatten(),0)

			Y[i,:,:,:] = I1

			if(not return_pose_vectors):
				yield ([X_src,X_pose_src,X_pose_tgt,X_mask_src,X_trans],Y)
			else:
				yield ([X_src,X_pose_src,X_pose_tgt,X_mask_src,X_trans,X_posevec_src,X_posevec_tgt],Y)
'''


def rand_scale(param):
    rnd = np.random.rand()
    return (param['scale_max'] - param['scale_min']) * rnd + param['scale_min']


def rand_rot(param):
    return (np.random.rand() - 0.5) * 2 * param['max_rotate_degree']


def rand_shift(param):
    shift_px = param['max_px_shift']
    x_shift = int(shift_px * (np.random.rand() - 0.5))
    y_shift = int(shift_px * (np.random.rand() - 0.5))
    return x_shift, y_shift


def rand_sat(param):
    min_sat = 1 - param['max_sat_factor']
    max_sat = 1 + param['max_sat_factor']
    return np.random.rand() * (max_sat - min_sat) + min_sat


def rand_augmentations(param):
    rflip = np.random.rand()
    rscale = rand_scale(param)
    rshift = rand_shift(param)
    rdegree = rand_rot(param)
    rsat = rand_sat(param)
    return rflip, rscale, rshift, rdegree, rsat


def augment(I, joints, rflip, rscale, rshift, rdegree, rsat, img_height, img_width):
    I, joints = aug_flip(I, rflip, joints)
    I, joints = aug_scale(I, rscale, joints)
    I, joints = aug_shift(I, img_width, img_height, rshift, joints)
    I, joints = aug_rotate(I, img_width, img_height, rdegree, joints)
    I = aug_saturation(I, rsat)
    return I, joints

"""
def center_and_scale_image(I, img_width, img_height, pos, scale, joints):
    I = cv2.resize(I, (0, 0), fx= scale, fy = scale)
    joints = joints * scale
    pdb.set_trace()
    x_offset = (img_width - 1.0) / 2.0 - pos[0] * scale
    y_offset = (img_height - 1.0) / 2.0 - pos[1] * scale

    T = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    I = cv2.warpAffine(I, T, (img_width, img_height))

    joints[:, 0] += x_offset
    joints[:, 1] += y_offset
    pdb.set_trace()
    return I, joints
"""
def center_and_scale_image(I, img_width, img_height, pos, scale, joints):
    scale = 0.355555
    I = cv2.resize(I, (0, 0), fx= scale, fy = scale)
    joints = joints * scale
    x_offset = (img_width - 1.0) / 2.0 - pos[0] * scale
    y_offset = (img_height - 1.0) / 2.0 - pos[1] * scale

    T = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    I = cv2.warpAffine(I, T, (img_width, img_height))

    joints[:, 0] += x_offset
    joints[:, 1] += y_offset

    return I, joints

def aug_joint_shift(joints, max_joint_shift):
    joints += (np.random.rand(joints.shape) * 2 - 1) * max_joint_shift
    return joints


def aug_flip(I, rflip, joints):
    if (rflip < 0.5):
        return I, joints

    I = np.fliplr(I)
    joints[:, 0] = I.shape[1] - 1 - joints[:, 0]

    right = [2, 3, 4, 8, 9, 10]
    left = [5, 6, 7, 11, 12, 13]

    for i in range(6):
        tmp = np.copy(joints[right[i], :])
        joints[right[i], :] = np.copy(joints[left[i], :])
        joints[left[i], :] = tmp

    return I, joints


def aug_scale(I, scale_rand, joints):
    I = cv2.resize(I, (0, 0), fx=scale_rand, fy=scale_rand)
    joints = joints * scale_rand
    return I, joints


def aug_rotate(I, img_width, img_height, degree_rand, joints):
    h = I.shape[0]
    w = I.shape[1]

    center = ((w - 1.0) / 2.0, (h - 1.0) / 2.0)
    R = cv2.getRotationMatrix2D(center, degree_rand, 1)
    I = cv2.warpAffine(I, R, (img_width, img_height))

    for i in range(joints.shape[0]):
        joints[i, :] = rotate_point(joints[i, :], R)

    return I, joints


def rotate_point(p, R):
    x_new = R[0, 0] * p[0] + R[0, 1] * p[1] + R[0, 2]
    y_new = R[1, 0] * p[0] + R[1, 1] * p[1] + R[1, 2]
    return np.array((x_new, y_new))


def aug_shift(I, img_width, img_height, rand_shift, joints):
    x_shift = rand_shift[0]
    y_shift = rand_shift[1]

    T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    I = cv2.warpAffine(I, T, (img_width, img_height))

    joints[:, 0] += x_shift
    joints[:, 1] += y_shift

    return I, joints


def aug_saturation(I, rsat):
    I *= rsat
    I[I > 1] = 1
    return I


def make_joint_heatmaps(height, width, joints, sigma, pose_dn):
    height = int(height / pose_dn)
    width = int(width / pose_dn)
    n_joints = joints.shape[0]
    var = sigma ** 2
    joints = joints / pose_dn

    H = np.zeros((height, width, n_joints))

    for i in range(n_joints):
        if (joints[i, 0] <= 0 or joints[i, 1] <= 0 or joints[i, 0] >= width - 1 or
                joints[i, 1] >= height - 1):
            continue

        H[:, :, i] = make_gaussian_map(width, height, joints[i, :], var, var, 0.0)

    return H


def make_gaussian_map(img_width, img_height, center, var_x, var_y, theta):
    xv, yv = np.meshgrid(np.array(range(img_width)), np.array(range(img_height)),
                         sparse=False, indexing='xy')

    a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
    b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
    c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

    return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                    2 * b * (xv - center[0]) * (yv - center[1]) +
                    c * (yv - center[1]) * (yv - center[1])))


def make_limb_masks(limbs, joints, img_width, img_height):
    n_limbs = len(limbs)
    mask = np.zeros((img_height, img_width, n_limbs))

    # Gaussian sigma perpendicular to the limb axis.
    sigma_perp = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 13]) ** 2

    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p[j, :] = [joints[limbs[i][j], 0], joints[limbs[i][j], 1]]

        if n_joints_for_limb == 4:
            p_top = np.mean(p[0:2, :], axis=0)
            p_bot = np.mean(p[2:4, :], axis=0)
            p = np.vstack((p_top, p_bot))

        center = np.mean(p, axis=0)

        sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 1.5])
        theta = np.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

        mask_i = make_gaussian_map(img_width, img_height, center, sigma_parallel, sigma_perp[i], theta)
        mask[:, :, i] = mask_i / (np.amax(mask_i) + 1e-6)

    return mask


def get_limb_transforms(limbs, joints1, joints2):
    n_limbs = len(limbs)

    Ms = np.zeros((2, 3, n_limbs))

    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p1 = np.zeros((n_joints_for_limb, 2))
        p2 = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p1[j, :] = [joints1[limbs[i][j], 0], joints1[limbs[i][j], 1]]
            p2[j, :] = [joints2[limbs[i][j], 0], joints2[limbs[i][j], 1]]

        tform = transformations.make_similarity(p2, p1, False)
        Ms[:, :, i] = np.array([[tform[1], -tform[3], tform[0]], [tform[3], tform[1], tform[2]]])

    return Ms
