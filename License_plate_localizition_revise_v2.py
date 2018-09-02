# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:42:00 2018

@author: Administrator
"""
import tensorflow as tf
import numpy as np

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import cv2
from matplotlib import pyplot as plt
import praseLabel2
import string
import pl_data

#cell_lenth = tf.constant([178, 120], dtype=tf.float32)
cell = [178, 120]
#cell_num = tf.constant([9, 10], dtype=tf.float32)
#boxes = tf.constant([100, 30, 125, 44], dtype=tf.float32)
boxes_ar = [100, 30, 125, 44]
#threshold = tf.constant([0.6, 0.6], dtype=tf.float32)
threshold = [0.6, 0.6]
#x_cell = 9
#y_cell = 10
batch_size = 12
lamda_coord = 5
lamda_noobj = 0.1
classes_num = 2
#160 * 50; 160 *160
anchor_num = 2

learn_rate = 0.001
empoch = 20000

FLAGS = None

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    var = sum(shape)
    initial = tf.truncated_normal(shape, stddev= 2.0 / var) #为了改善可能出现的梯度消失和梯度爆炸问题，让方差为 2/n
    #initial = tf.cast(initial, tf.int8)
    return tf.Variable(initial, dtype = tf.float32)

def bias_variable(shapes):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1,shape = shapes)
    #initial = tf.cast(initial, tf.int8)
    return tf.Variable(initial, dtype = tf.float32)

def zf_net(input_tensor):
    """
    build a zf net 
    Args:
        input_tensor: the input image tensor with the shape [batch, 1200, 1600, 3]
    Return:
        output_tensor: the output tenosr with the shape [batch, 68, 90, 256]
    """
    # First conv layer 7*7*96 stride = 2 padding = same
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([7, 7, 3, 96])
        b_conv1 = bias_variable([96])
        A_conv1 = tf.nn.conv2d(input_tensor, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        Z_conv1 = tf.nn.relu(A_conv1)
    tf.summary.histogram('w_conv1', W_conv1)
    tf.summary.histogram('b_conv1', b_conv1)
    # output shape [batch, 600, 800, 96]
        
    # First max_pooling layer 3*3 s = 2 padding = same
    with tf.name_scope('pool1'):
        Z_pool1 = tf.nn.max_pool(Z_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # output shape [batch, 300, 400, 96]
        
    # Second conv layer 5*5*256 2 same
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 96, 256])
        b_conv2 = bias_variable([256])
        A_conv2 = tf.nn.conv2d(Z_pool1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
        Z_conv2 = tf.nn.relu(A_conv2)
    # output shape [batch, 150, 200, 256]
        
    # Second max_pooling layer 3*3 s = 2 padding = same
    with tf.name_scope('pool2'):
        Z_pool2 = tf.nn.max_pool(Z_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    #output shape [batch, 75, 100, 256]        
        
    # Third conv layer 3*3*384 1 same
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 256, 384])
        b_conv3 = bias_variable([384])
        A_conv3 = tf.nn.conv2d(Z_pool2, W_conv3, strides=[1, 2, 2, 1], padding='SAME') + b_conv3
        Z_conv3 = tf.nn.relu(A_conv3)
    # output shape [batch, 38, 50, 384]
    
    # Third max_pooling layer 3*3 s = 2 padding = same
    with tf.name_scope('pool3'):
        Z_pool3 = tf.nn.max_pool(Z_conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    #output shape [batch, 19, 25, 384]
          
    # Fourth conv layer 3*3*384 1 same
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 384, 384])
        b_conv4 = bias_variable([384])
        A_conv4 = tf.nn.conv2d(Z_pool3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4
        Z_conv4 = tf.nn.relu(A_conv4)
    #output shape [batch, 19, 25, 384]
    
    # Fifth conv layer 2*3*256 1 same
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([2, 3, 384, 256])
        b_conv5 = bias_variable([256])
        A_conv5 = tf.nn.conv2d(Z_conv4, W_conv5, strides=[1, 2, 3, 1], padding='SAME') + b_conv5
        Z_conv5 = tf.nn.relu(A_conv5)
    #output shape [batch, 10, 9, 256]
    
    # Sixth conv layer 1*1*256,use conv to achieve sliding window
    with tf.name_scope('fc6'):
        W_conv6 = weight_variable([1, 1, 256, 256])
        b_conv6 = bias_variable([256])
        A_conv6 = tf.nn.conv2d(Z_conv5, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6
        Z_conv6 = tf.nn.relu(A_conv6)
    # output shape [batch, 10, 9, 256]
    
    # Seventh conv layer 1*1*128
    with tf.name_scope('fv7'):
        W_conv7 = weight_variable([1, 1, 256, 128])
        b_conv7 = bias_variable([128])
        A_conv7 = tf.nn.conv2d(Z_conv6, W_conv7, strides=[1, 1, 1, 1], padding='SAME') + b_conv7
        Z_conv7 = tf.nn.relu(A_conv7)
    #output shape [batch, 10, 9, 128]
    
    # Eighth conv layer 1*1*2
    with tf.name_scope('fv8'):
        W_conv8 = weight_variable([1, 1, 128, 2])
        b_conv8 = bias_variable([1])
        A_conv8 = tf.nn.conv2d(Z_conv7, W_conv8, strides=[1, 1, 1, 1], padding='SAME') + b_conv8
        #A_conv8_cast = tf.cast(A_conv8, tf.float32)
        Z_conv8 = tf.nn.softmax(A_conv8)
    #output shape [batch, 10, 9, 2] (background or license plate)
    
    # Nineth conv layer 1*1*4
    with tf.name_scope('fv9'):
        W_conv9 = weight_variable([1, 1, 128, 10])
        b_conv9 = bias_variable([1])
        A_conv9 = tf.nn.conv2d(Z_conv7, W_conv9, strides=[1, 1, 1, 1], padding='SAME') + b_conv9
       # A_conv9_cast = tf.cast(A_conv9, tf.float32)
        Z_conv9 = tf.sigmoid(A_conv9)
        #output shape [batch, 10, 9, 10]
    '''
    temp_list = [1,]
    temp_list.extend(boxes)
    temp_list = temp_list * anchor_num
    bounding = tf.constant(temp_list, dtype = tf.float32)
    pre_box = tf.multiply(Z_conv9, bounding) # p_w = b_w * e ^ t_w... 
    '''
        
    return Z_conv8, Z_conv9

def iou(bbx1, bbx2):
    """
    calculate iou of bbx1 and bbx2
    Args:
        bbx1 or bbx2 : tensor with 1D shpae [x,y,w,h] (x,y) denote absolute center, (w,h) also absolute
    """
    s_all = bbx1[:, 2] * bbx1[:, 3] + bbx2[:, 2] * bbx2[:, 3]
    #print(s_all)
    #将中心坐标转变为左上和右下坐标
    '''
    bbx1 = tf.concat([tf.subtract(np.array(bbx1[0:2]), tf.divide(np.array(bbx1[2:4]), 2)), 
                     tf.add(np.array(bbx1[0:2]), tf.divide(np.array(bbx1[2:4]), 2))], 0)
    bbx2 = tf.concat([tf.subtract(np.array(bbx2[0:2]), tf.divide(np.array(bbx2[2:4]), 2)), 
                     tf.add(np.array(bbx2[0:2]), tf.divide(np.array(bbx2[2:4]), 2))], 0)
    '''
    bbx1 = tf.concat([tf.subtract(bbx1[0:2], tf.divide(bbx1[2:4], 2)), tf.add(bbx1[0:2], tf.divide(bbx1[2:4], 2))], 0)
    bbx2 = tf.concat([tf.subtract(bbx2[0:2], tf.divide(bbx2[2:4], 2)), tf.add(bbx2[0:2], tf.divide(bbx2[2:4], 2))], 0)
    lu = tf.maximum(bbx1[0:2], bbx2[0:2])
    rd = tf.minimum(bbx1[2:4], bbx2[2:4])
    #print(bbx1,bbx2)
    intersection = rd - lu
    inter_square = intersection[0] * intersection[1]
    mask = tf.cast(intersection[0] > 0, tf.float32) * tf.cast(intersection[1] > 0, tf.float32)
    inter_square = tf.multiply(inter_square, mask)
    return inter_square / (s_all - inter_square + 1e-6)
    #return s_all
    
def batch_iou(bbx1, bbx2):
    """
    calculate iou of bbx1 and bbx2
    Args:
        bbx1 or bbx2 : tensor with 2D shpae batch * [x,y,w,h] (x,y) denote absolute center, (w,h) also absolute
    """
    s_all = bbx1[:, 2] * bbx1[:, 3] + bbx2[:, 2] * bbx2[:, 3]
    bbx_process1 = tf.concat([(bbx1[:, 0:2] - bbx1[:, 2:4] / 2), (bbx1[:, 0:2] + bbx1[:, 2:4] / 2)], 1)
    bbx_process2 = tf.concat([(bbx2[:, 0:2] - bbx2[:, 2:4] / 2), (bbx2[:, 0:2] + bbx2[:, 2:4] / 2)], 1)
    lu = tf.maximum(bbx_process1[:, 0:2], bbx_process2[:, 0:2])
    rd = tf.minimum(bbx_process1[:, 2:4], bbx_process2[:, 2:4])
    intersection = rd - lu
    inter_square = intersection[:, 0] * intersection[:, 1]
    mask = tf.cast(intersection[:, 0] > 0, tf.float32) * tf.cast(intersection[:, 0] > 0, tf.float32)
    inter_square = inter_square * mask
    return inter_square / (s_all - inter_square + 1e-6)#[batch]

def count_loss(lp_classes, lp_location, gt_tensor):
    """
    Args:
        input_tensor: [batch, 1200, 1600, 3]
        gt_tensor: [batch, 5]
    return:
        Cross_entropy_mean
    """
    
    #lp_classes, lp_location = zf_net(input_tensor)#[10,9,2],[10,9,10]
    #lp_res = tf.arg_max(lp_classes, 3)
    #gt_coord,gt_norm = gt_toGride(gt_tensor) # batch * (x,y)int32,
    #[confidence,norm_x,norm_y,norm_w1,norm_h1,norm_w2,norm_h2]float32
 
    gt_class = tf.cast(gt_tensor[:, 0], tf.int32)
    gt_cls = tf.one_hot(gt_class, classes_num, dtype = tf.float32)
    pre_p, predicts = predict_net_forloss(lp_classes, lp_location)
    batchs_iou = batch_iou(predicts[:, 1:5], gt_tensor[:, 1:5])
    C1 = tf.nn.l2_loss(predicts[:, 1:3] - gt_tensor[:, 1:3])
    C2 = tf.nn.l2_loss(tf.sqrt(predicts[:, 3:5]) - tf.sqrt(gt_tensor[:, 3:5]))
    C3 = tf.nn.l2_loss(predicts[:, 0] - batchs_iou)
    C4 = tf.nn.l2_loss(lp_location[:, :, :, 0]) + tf.nn.l2_loss(lp_location[:, :, :, 5]) - tf.nn.l2_loss(predicts[0])
    C5 = tf.nn.l2_loss(pre_p - gt_cls)
    tf.summary.scalar('C1', C1)
    tf.summary.scalar('C2', C2)
    tf.summary.scalar('C3', C3)
    tf.summary.scalar('C4', C4)
    tf.summary.scalar('C5', C5)
    cross_entropy_mean = lamda_coord * (C1 + C2) + C3 + lamda_noobj * C4 + C5
    tf.summary.scalar('Loss', cross_entropy_mean)    
    return cross_entropy_mean

def predict_net_forloss(z8, z9):
    """
    Give the prediction based on the z8, z9.
    Args:
        z8 : a tensor with shape [batch, 10, 9, 2] #classes
        z9 : a tensor with shape [batch, 10, 9 ,10] #bbx
    Return:
        prediction: a tensor with shape [batch, 5]  [p, x, y, w, h]
    """
    temp_confidence = tf.concat([z9[:, :, :, 0:1], z9[:, :, :, 5:6]], 3)
    batch = batch_size#z8.shape[0]
    temp_confidence = tf.reshape(temp_confidence, (batch, -1))
    max_index = tf.argmax(temp_confidence, 1) #max index [batch]
    max_index_re = tf.reshape(max_index, [batch, 1])
    #temp_list = np.array(value=range(batch),dtype=np.int64) 
    temp_index = tf.constant(list(range(batch)), shape=[batch, 1],dtype=tf.int64)
    last_index = tf.concat([temp_index, max_index_re], 1) #[batch, 2]
    '''
    pre_p = tf.reshape(z8, [batch, -1, 2]) # [batch, 180, 5]
    pre_pl = pre_p[:, :, 1:2] - 1.0 #取车牌相对概率
    pl_paddings = tf.constant([[0, 0], [0, 0], [0, 4]])
    pre_pl_pad = tf.pad(pre_pl, pl_paddings, "CONSTANT") + 1.0 #[BATCH, 180, 5]
    '''
    pre_norm = tf.reshape(z9, (batch, -1, 2, 5))# [batch, 90, 2, 5] turn output tensor to prediction of init image
    mask1 = tf.constant([0, 0, 0, 1, 1], dtype=tf.float32)
    mask2 = tf.constant([1, 1, 1, 0, 0], dtype=tf.float32)
    maskBoundbox = tf.constant([[0, 0, 0, boxes_ar[0], boxes_ar[1]],[0, 0, 0, boxes_ar[2], boxes_ar[3]]], dtype=tf.float32)
    pre_bound = tf.exp(pre_norm * mask1) * maskBoundbox + pre_norm * mask2
    
    pre_box = tf.reshape(pre_bound,(batch, -1, 5)) # [batch,180,5]

    pre_init = tf.gather_nd(pre_box, last_index) #[batch,5]
    
    max_index_p = tf.cast(max_index_re / 2, tf.int64)
    last_index_p = tf.concat([temp_index, max_index_p], 1)
    z8_re = tf.reshape(z8, (batch, -1, 2)) #[batch, 90, 2]
    pre_p = tf.gather_nd(z8_re, last_index_p) #[batch, 2]#取车牌相对概率
    
    return pre_p, pre_init
    
def predict_net(z8, z9):
    """
    Give the prediction based on the z8, z9.
    Args:
        z8 : a tensor with shape [batch, 10, 9, 2] #classes
        z9 : a tensor with shape [batch, 10, 9 ,10] #bbx
    Return:
        prediction: a tensor with shape [batch, 5]  [c*p, x, y, w, h]
    """
    temp_confidence = tf.concat([z9[:, :, :, 0:1], z9[:, :, :, 5:6]], 3)
    batch = batch_size#z8.shape[0]
    temp_confidence = tf.reshape(temp_confidence, (batch, -1)) # [batch,180]
    max_index = tf.argmax(temp_confidence, 1) #max index [batch]
    max_index_re = tf.reshape(max_index, [batch, 1])
    #temp_list = np.array(value=range(batch),dtype=np.int64) 
    temp_index = tf.constant(list(range(batch)), shape=[batch, 1],dtype=tf.int64)
    last_index = tf.concat([temp_index, max_index_re], 1) #[batch, 2]
    '''
    pre_p = tf.reshape(z8, [batch, -1, 2]) # [batch, 180, 5]
    pre_pl = pre_p[:, :, 1:2] - 1.0 #取车牌相对概率
    pl_paddings = tf.constant([[0, 0], [0, 0], [0, 4]])
    pre_pl_pad = tf.pad(pre_pl, pl_paddings, "CONSTANT") + 1.0 #[BATCH, 180, 5]
    '''
    pre_norm = tf.reshape(z9, (batch, -1, 2, 5))# [batch, 90, 2, 5] turn output tensor to prediction of init image
    mask1 = tf.constant([0, 0, 0, 1, 1], dtype=tf.float32)
    mask2 = tf.constant([1, 1, 1, 0, 0], dtype=tf.float32)
    maskBoundbox = tf.constant([[0, 0, 0, boxes_ar[0], boxes_ar[1]],[0, 0, 0, boxes_ar[2], boxes_ar[3]]], dtype=tf.float32)
    pre_bound = tf.exp(pre_norm * mask1) * maskBoundbox + pre_norm * mask2
    
    pre_box = tf.reshape(pre_bound,(batch, -1, 5)) # [batch,180,5]
    pre_init = tf.gather_nd(pre_box, last_index) #[batch,5]
    
    max_index_p = tf.cast(max_index_re / 2, tf.int64)
    last_index_p = tf.concat([temp_index, max_index_p], 1)
    z8_re = tf.reshape(z8, (batch, -1, 2)) #[batch, 90, 2]
    pre_p = tf.gather_nd(z8_re, last_index_p) #[batch, 2]#取车牌相对概率
    pre_pl = pre_p[:, 1:2] - 1.0 
    pl_paddings = tf.constant([[0, 0], [0, 4]])
    pre_pl_pad = tf.pad(pre_pl, pl_paddings, "CONSTANT") + 1.0 #[BATCH,  5]
    pre_init = pre_init * pre_pl_pad # c * p(c/object)
    max_index_cor = tf.squeeze(max_index_p)
    pre_y = tf.div(max_index_cor, 9)
    pre_x = tf.mod(max_index_cor, 9)
    pre_coord = tf.stack([pre_x, pre_y])
    paddings = tf.constant([[1, 2], [0, 0]]) 
    mask_coord = tf.cast(tf.pad(pre_coord, paddings, "CONSTANT"),tf.float32)#[batch, 5]
    mask_coord = tf.transpose(mask_coord)
    mask3 = tf.constant([1, cell[0], cell[1], 1, 1], dtype=tf.float32)
    prediction = (pre_init + mask_coord) * mask3
    
    return prediction #[batch, 5]

def precise_loss(label, prediction):
    """
    Args:
        label: [batch, 5]
        prediction: [batch, 5]
    return:
        count:[], tf.int64
    """
    class_p = tf.cast(prediction[:, 0] >= threshold[0], tf.float32)
    batchs_iou = batch_iou(label[:, 1:5], prediction[:, 1:5])
    compare_temp = (batchs_iou >= threshold[1])
    compare_temp = tf.cast(compare_temp, tf.float32)
    precise = tf.reduce_mean(compare_temp * class_p)
    """
    for i in range(batch):
        iou_pre = iou(label[i], prediction[i])
        compare_temp = (iou_pre >= thres[1]) 
        compare_temp = tf.cast(compare_temp, tf.int64)
        count = count + compare_temp * class_p[i]
    """  
    return precise

def shullf_name(Num):
    
    return ''.join(random.choice(string.ascii_uppercase + 
                                 string.ascii_lowercase+ string.digits) for _ in range(Num))
    
def test_save_image(image, label, save_path):
    """ draw the label in image
    Args:
        image: [batch, 1200, 1600, 3]
        label: [batch, 5]-[c, x, y, w, h]
    """
    #image = tf.reverse(image, [3])
    batch = batch_size#image.shape[0]
    for i in range(batch):
        image_i = praseLabel2.drawRect(image[i], label[i])
        image_name = os.path.join(save_path,shullf_name(16) + '.jpg')
        while os.path.exists(image_name):
            image_name = os.path.join(save_path,shullf_name(16) + '.jpg')
        praseLabel2.cv_write(image_name, image_i)

#def add_train(is_training):
def name_count_tfrecord(tf_path):
    train_path = os.path.join(FLAGS.tfrecord_dir, 'training.tfrecord')
    test_path = os.path.join(FLAGS.tfrecord_dir, 'validation.tfrecord')
    valid_path = os.path.join(FLAGS.tfrecord_dir, 'testing.tfrecord')
    outputpath = [train_path, test_path, valid_path]
    count = []
    #small_count = 0
    for file in outputpath:
        count.append(len(list(tf.python_io.tf_record_iterator(file))))
        
    return outputpath, count

def get_data(name, batch, if_random, tf_record_path):
    tf_filename = os.path.join(tf_record_path, name + '.tfrecord')
    images, labels = pl_data.read_tfrecord(tf_filename,batch, if_random)
    return images, labels
        
def train_test(input_images, input_labels):
    z8, z9 = zf_net(input_images)#[10,9,2],[10,9,10]
    cross_entropy = count_loss(z8, z9, input_labels) #[]
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    train_step = optimizer.minimize(cross_entropy)
    
    prediction = predict_net(z8, z9)#[batch, 5]
    precise = precise_loss(input_labels, prediction)

    return cross_entropy, train_step, prediction, precise   

def process_image(images):
    images = tf.image.convert_image_dtype(images,tf.int8)
    image = tf.reverse(images, [3])
    return image
    
def add_train(train_path):
    
    train_path = os.path.join(train_path, 'training.tfrecord')
    train_images, train_labels = pl_data.read_tfrecord(train_path, batch_size, True)
    #pre_class, pre_bbx = zf_net(train_images) #[batch, 10, 9, 2], [batch, 10, 9, 10]
    cross_entropy = count_loss(train_images, train_labels) #[]
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    train_step = optimizer.minimize(cross_entropy)
    
    return cross_entropy, train_step

def add_valid(valid_path, valid_size):
    
    valid_path = os.path.join(valid_path, 'validation.tfrecord')        
    valid_images, valid_labels = pl_data.read_tfrecord(valid_path, valid_size, True)
    pre_class, pre_bbx = zf_net(valid_images)
    predic = predict_net(pre_class, pre_bbx)
    count = precise_loss(valid_labels, predic)
    return count / valid_size

def add_test(test_path, test_size, is_random):
    test_path = os.path.join(test_path, 'testing.tfrecord')
    test_images, test_labels = pl_data.read_tfrecord(test_path, test_size, is_random)
    pre_class, pre_bbx = zf_net(test_images)
    predic = predict_net(pre_class, pre_bbx)
    test_precise = precise_loss(test_labels, predic) / test_size
    images = tf.image.convert_image_dtype(test_images,tf.int8)
    image = tf.reverse(images, [3])
    return test_precise, image, predic

def add_debug():
    path =  os.path.join(FLAGS.tfrecord_dir, 'training.tfrecord')
    train_images, train_labels = pl_data.read_tfrecord(path, batch_size, True)
    pre_class, pre_bbx = zf_net(train_images)
    
    return pre_class, pre_bbx
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.reset_default_graph()    
    tf_path  = FLAGS.tfrecord_dir
    tfrecord_name = ['training', 'validation', 'testing']
    save_test = True
    image_tensor = tf.placeholder(tf.float32,[None, 1200, 1600, 3])
    label_tensor = tf.placeholder(tf.float32, [None, 5])
    cross_entropy, train_step, prediction, precise = train_test(image_tensor, label_tensor)
    image_train, label_train = get_data(tfrecord_name[0], batch_size, True, tf_path)
    image_valid, label_valid = get_data(tfrecord_name[1], batch_size, True, tf_path)  
    image_test, label_test = get_data(tfrecord_name[2], batch_size, True, tf_path)
    visual_image = process_image(image_test)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:  
        train_writer = tf.summary.FileWriter('log_8_26',sess.graph)             
        sess.run(init)
        #saver.restore(sess, os.path.join(FLAGS.var_dir, 'model.ckpt'))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
                             
            for i in range(empoch+1):
                image_train_np, label_train_np = sess.run([image_train, label_train])
                if(i % 200 == 99):
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    cross_entropy_value, _, precise_value, summary = sess.run([cross_entropy, train_step, precise, merged],
                                feed_dict={image_tensor: image_train_np, label_tensor: label_train_np},
                                options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                    train_writer.add_summary(summary, i)
                else:            
                    cross_entropy_value, _, precise_value, summary = sess.run([cross_entropy, train_step, precise, merged],
                                feed_dict={image_tensor: image_train_np, label_tensor: label_train_np})
                    train_writer.add_summary(summary, i)
                if(i % 50 == 0):
                    tf.logging.info('%s: Step %d: Train loss = %f, Train_accuracy = %f' %
                                    (datetime.now(), i, cross_entropy_value, precise_value))
                if(i % 200 == 0):
                    accuracy_all = 0;
                    for i in range(5):
                        image_valid_np, label_valid_np = sess.run([image_valid, label_valid])
                        cross_entropy_value, precise_value = sess.run([cross_entropy, precise],
                                feed_dict={image_tensor: image_valid_np, label_tensor: label_valid_np})
                        accuracy_all += precise_value
                        tf.logging.info('%s: Step %d: Valid loss = %f, Test_accuracy = %f' %
                                    (datetime.now(), i, cross_entropy_value, precise_value))
                    tf.logging.info('%s: Step %d : the mean accuracy = %f'%
                                    (datetime.now(), i, accuracy_all / 5))
            
            train_writer.close()
            for i in range(5):
                image_test_np, label_test_np, images_np = sess.run([image_test, label_test, visual_image])
                prediction_value, precise_value = sess.run([prediction, precise],
                    feed_dict={image_tensor: image_test_np, label_tensor: label_test_np})
                tf.logging.info('%s: Test image num : %d  Test accuracy = %f' %
                                (datetime.now(), batch_size, precise_value))
                print(images_np.shape, prediction_value.shape)
                if save_test:
                    test_save_image(images_np, prediction_value, FLAGS.save_test_image)
            
            model_save_path = saver.save(sess, os.path.join(FLAGS.model_dir, 'model.ckpt'))
            tf.logging.info('%s: Model saved in path: %s' %
                            (datetime.now(), model_save_path))   
        except tf.errors.OutOfRangeError:
            print("out of range of threads")
        finally:
            coord.request_stop()
        
        coord.join(threads)
        
    
                
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type = str,
        default = r'G:\GraduateStudy\Smoke Recognition\Newdata\Train',
        help = 'Path to folders of images.'
    )
    parser.add_argument(
        '--test_dir',
        type = str,
        default = r'G:\GraduateStudy\Smoke Recognition\Newdata\Test',
        help = 'Path to folders of images.'
    )
    parser.add_argument(
        '--tfrecord_dir',
        type = str,
        default = r'F:\Pr_ml\data\tf_record',
        help = 'Path to folder to save tfrecord.'
    )
    parser.add_argument(
        '--model_dir',
        type = str,
        default = r'F:\Pr_ml\data\save_model',
        help = 'Path to folder to save model.'
    )         
    parser.add_argument(
        '--save_test_image',
        type = str,
        default = r'F:\Pr_ml\data\save_test',
        help = 'path to folders of saving tested images'
    )       
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)            