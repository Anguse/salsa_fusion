from __future__ import print_function, division
import os.path
from config import *
from utils import *
from model import *
from segmenter import SegmenterNet
import numpy as np
import matplotlib.image as pimg
import matplotlib.pyplot as plt
from glob import glob
import sys
import time


def test_with_bev_images(net, img):

    with tf.Session(graph=net.graph) as sess:
        net.initialize_vars(sess)

        print("loading bev img " + str(img))
        lidar_data = np.load(img)
        netInput = lidar_data[:, :, 0:4]  # mean, max, ref, den
        netInput = netInput.astype('float32') / 255

        pred_img = net.predict_single_image(input_img=netInput, session=sess)
        print('predicted image shape: ', pred_img.shape, ' type: ',  pred_img.dtype,
              ' min val: ',  pred_img.min(), ' max val: ', pred_img.max())

        showPredImg(pred_img)


def test_with_point_cloud(net, cfg, cloud_path):

    # load pc
    pointCloud = np.load(cloud_path)

    # define the region of interest for bird eye view image generation
    pc2img = PC2ImgConverter(imgChannel=cfg.IMAGE_CHANNEL, xRange=[0, 50], yRange=[-6, 12], zRange=[-10, 8],
                             xGridSize=0.2, yGridSize=0.3, zGridSize=0.3, maxImgHeight=cfg.IMAGE_HEIGHT,
                             maxImgWidth=cfg.IMAGE_WIDTH, maxImgDepth=64)

    bevImg, bevCloud = pc2img.getBEVImage(pointCloud)
    bevImg = bevImg.astype('float32') / 255
    print('bird eye view image shape: ', bevImg.shape)

    with tf.Session(graph=net.graph) as sess:
        net.initialize_vars(sess)

        pred_img = net.predict_single_image(input_img=bevImg, session=sess)
        print('predicted image shape: ', pred_img.shape, ' type: ', pred_img.dtype, ' min val: ', pred_img.min(),
              ' max val: ', pred_img.max())
        roadCloud, vehicleCloud = pc2img.getCloudsFromBEVImage(
            pred_img, bevCloud, postProcessing=True)

        showPredImg(pred_img)


def test_with_depth(net, cfg, depth_path):

    # load depth
    depth = np.load(depth_path)
    label = depth[:, :, 1]
    depth = depth[:, :, 0:1]
    pimg.imsave('../preds/depth_orig.png', depth[:, :, 0])
    pimg.imsave('../preds/depth_label.png', label)
    #depth = np.expand_dims(depth, axis=3)
    depth_img = depth.astype('float32') / 255

    with tf.Session(graph=net.graph) as sess:
        net.initialize_vars(sess)

        pred_img = net.predict_single_image(input_img=depth_img, session=sess)
        tps, fps, fns = net.evaluate_iou(label, pred_img, cfg.NUM_CLASS)
        #iou = sess.run(net.evaluation)
        iou = 0
        tps = np.array(tps)
        fps = np.array(fps)
        fns = np.array(fns)
        epsilon = 1e-12
        ious = tps.astype(np.float) / (tps + fns + fps + epsilon)
        precision = tps.astype(np.float) / (tps + fps + epsilon)
        recall = tps.astype(np.float) / (tps + fns + epsilon)

        mean_ious = np.mean(ious, axis=0)
        mean_prec = np.mean(precision, axis=0)
        mean_recall = np.mean(recall, axis=0)

        s = "TEST IOU: {: 3.3f} TEST IOU: {: 3.3f}"
        print(s.format(iou, np.mean(ious)))

        for i in range(0, cfg.NUM_CLASS):
            s = cfg.CLASSES[i] + " PREC: {: 3.3f} " + cfg.CLASSES[i] + \
                " REC: {: 3.3f} " + cfg.CLASSES[i] + " IOU: {: 3.3f}"
            print(s.format(precision[i], recall[i], ious[i]))

        print('predicted image shape: ', pred_img.shape, ' type: ', pred_img.dtype, ' min val: ', pred_img.min(),
              ' max val: ', pred_img.max())

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax1.axis('off')
        ax1.imshow(depth[:, :, 0])
        ax1.set_title('Original image')
        ax2.axis('off')
        ax2.imshow(label)
        ax2.set_title('Ground truth')
        ax3.axis('off')
        ax3.imshow(pred_img)
        ax3.set_title('Prediction')
        plt.savefig('../preds/pred_plot.png', bbox_inches='tight')

        pimg.imsave('../preds/depth_pred.png', pred_img)
        # showPredImg(pred_img)
        #np.save('../preds/depth_pred.npy', pred_img)


def test_with_depth_batches(net, cfg, test_path, iterations):
    with tf.Session(graph=net.graph) as sess:
        net.initialize_vars(sess)

        scores = []
        for i in range(iterations):
            # Generate batches
            test_batches, n_test_samples = generate_lidar_batch_function(
                test_path, channel_nbr=cfg.IMAGE_CHANNEL, class_nbr=cfg.NUM_CLASS, loss_weights=cfg.CLS_LOSS_WEIGHTS, augmentation=cfg.DATA_AUGMENTATION)

            # Num batches per epoch
            n_batches = int(np.ceil(n_test_samples / cfg.BATCH_SIZE))

            # Iterate through each mini-batch
            # for step in range(n_batches):
            #    X_batch, Y_batch, W_batch = next(test_batches(cfg.BATCH_SIZE))
            #    batch_data = X_batch
            #    preds = net.predict(batch_data, sess)

            # Evaluate
            test_iou, test_loss, test_ious, test_precs, test_recalls = net.evaluate(
                test_batches, n_test_samples, sess)
            # print scores
            net.print_evaluation_scores(
                test_iou, test_loss, test_ious, test_precs, test_recalls, tag="Testing")
            score = np.array([test_ious, test_precs, test_recalls])
            scores.append(score)
        scores = np.array(scores)
        return scores, n_test_samples


def showPredImg(img=[]):

    plt.imshow(img)
    plt.axis('off')

    plt.show()


def main():

    # get config params
    cfg = depth_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)
    print("creating network model using gpu " + str(cfg.GPU))

    # get trained model checkpoints
    model_ckp_name = "../logs/dataset_newnew/depth/model.chk-100"

    # load the trained model
    net = SegmenterNet(cfg, model_ckp_name)

    # testing with the already created bird eye view images
    #test_bev_img = "../data/test_bev_image.npy"
    #test_with_bev_images(net, test_bev_img)

    # testing by using the raw point cloud data
    #test_path = "/data/tmp/hl_data/dataset/depth/test/"
    test_path = "/data/tmp/hl_data/validation/depth/test/all/"
    depth_images = list(glob(os.path.join(test_path, '*.npy')))
    np.random.shuffle(depth_images)
    # print(depth_images)
    #nr = depth_images.index(test_path+'val1_depth_labeled48.npy')
    nr = 0
    input_image = depth_images[nr]
    test_with_depth(net, cfg, input_image)
    print(input_image)

    '''
    # np.random.shuffle(depth_images)
    start = time.time()
    iterations = 100
    score, n_test_samples = test_with_depth_batches(
        net, cfg, test_path, iterations)
    print("wall IoU: %3f, PREC: %3f, REC: %3f" % (np.mean(
        score[:, 0, 0]), np.mean(score[:, 1, 0]), np.mean(score[:, 2, 0])))
    print("ground IoU: %3f, PREC: %3f, REC: %3f" % (np.mean(
        score[:, 0, 1]), np.mean(score[:, 1, 1]), np.mean(score[:, 2, 1])))
    print("obstacle IoU: %3f, PREC: %3f, REC: %3f" % (np.mean(
        score[:, 0, 2]), np.mean(score[:, 1, 2]), np.mean(score[:, 2, 2])))
    print("car IoU: %3f, PREC: %3f, REC: %3f" % (np.mean(
        score[:, 0, 3]), np.mean(score[:, 1, 3]), np.mean(score[:, 2, 3])))
    print("unknown IoU: %3f, PREC: %3f, REC: %3f" % (np.mean(
        score[:, 0, 4]), np.mean(score[:, 1, 4]), np.mean(score[:, 2, 4])))
    print("average IoU: %3f, PREC: %3f, REC: %3f" % (np.mean(np.mean(score[:, 0, :], axis=1)), np.mean(
        np.mean(score[:, 1, :], axis=1)), np.mean(np.mean(score[:, 2, :], axis=1))))
    end = time.time()
    print("%3f seconds per image" % ((end-start)/(iterations*n_test_samples)))
    '''

    #nr = int(sys.argv[1])
    #image = depth_images[nr]
    # print(image)
    #test_with_depth(net, cfg, image)

    return


if __name__ == '__main__':

    main()
