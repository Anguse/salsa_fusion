# TF SHORTCUTS
import tensorflow as tf
import tensorflow.contrib as tc
conv2d_layer = tc.layers.conv2d
conv2d_trans_layer = tc.layers.conv2d_transpose
conv1d_layer = tf.nn.conv1d
conv1d_trans_layer = tc.nn.conv1d_transpose
concat = tf.keras.layers.Concatenate

leakyRelu = tf.nn.leaky_relu
maxpool_layer = tc.layers.max_pool2d
dropout_layer = tf.layers.dropout
batchnorm = tc.layers.batch_norm


def resBlock(input_layer, filter_nbr, dropout_rate, kernel_size=(3, 3), stride=1, layer_name="rb", training=True,
             pooling=True, repetition=1):
    with tf.variable_scope(layer_name):

        resA = input_layer

        for i in range(repetition):
            shortcut = conv2d_layer(resA, filter_nbr, kernel_size=(1, 1), stride=stride, activation_fn=leakyRelu,
                                    scope=layer_name + '_s_%d' % (i + 0))

            resA = conv2d_layer(resA, filter_nbr, kernel_size, normalizer_fn=batchnorm,
                                activation_fn=leakyRelu,
                                normalizer_params={'is_training': training},
                                scope=layer_name + '_%d_conv1' % (i + 0))

            resA = conv2d_layer(resA, filter_nbr, kernel_size, normalizer_fn=batchnorm,
                                activation_fn=leakyRelu,
                                normalizer_params={'is_training': training},
                                scope=layer_name + '_%d_conv2' % (i + 0))

            resA = tf.add(resA, shortcut)

        if pooling:
            resB = dropout_layer(resA, rate=dropout_rate, name="dropout")
            resB = maxpool_layer(resB, (2, 2), padding='same')

            print(str(layer_name) + str(resB.shape.as_list()))
            return resB, resA
        else:
            resB = dropout_layer(resA, rate=dropout_rate, name="dropout")
            print(str(layer_name) + str(resB.shape.as_list()))
            return resB


def resBlock1d(input_layer, filter_nbr, dropout_rate, kernel_size=(3, 3), stride=1, layer_name="rb", training=True,
               pooling=True, repetition=1):
    with tf.variable_scope(layer_name):

        resA = input_layer

    for i in range(repetition):
        # scope=layer_name + '_s_%d' % (i + 0)
        # kernel_size=(1, 1)
        # activation_fn=leakyRelu
        shortcut = conv1d_layer(
            resA, filter_nbr, stride=stride, padding='SAME')

        # scope=layer_name + '_%d_conv1' % (i + 0)
        # kernel_size
        # activation_fn=leakyRelu
        # normalizer_fn=batchnorm
        # normalizer_params={'is_training': training}
        resA = conv1d_layer(
            resA, filter_nbr, stride=stride, padding='VALID')

        # scope=layer_name + '_%d_conv2' % (i + 0)
        # kernel_size
        # normalizer_fn=batchnorm
        # activation_fn=leakyRelu
        # normalizer_params={'is_training': training}
        resA = conv1d_layer(
            resA, filter_nbr, stride=stride, padding='VALID')

        resA = tf.add(resA, shortcut)

    if pooling:
        resB = dropout_layer(resA, rate=dropout_rate, name="dropout")
        resB = maxpool_layer(resB, (2, 2), padding='same')

        print(str(layer_name) + str(resB.shape.as_list()))
        return resB, resA
    else:
        resB = dropout_layer(resA, rate=dropout_rate, name="dropout")
        print(str(layer_name) + str(resB.shape.as_list()))
        return resB


def upBlock(input_layer, skip_layer, filter_nbr, dropout_rate, kernel_size=(3, 3), layer_name="dec", training=True):
    with tf.variable_scope(layer_name + "_up"):
        upA = conv2d_trans_layer(input_layer, filter_nbr, kernel_size, 2, normalizer_fn=batchnorm,
                                 activation_fn=leakyRelu,
                                 normalizer_params={'is_training': training}, scope="tconv")
        upA = dropout_layer(upA, rate=dropout_rate, name="dropout")

    with tf.variable_scope(layer_name + "_add"):
        upB = tf.add(upA, skip_layer, name="add")
        upB = dropout_layer(upB, rate=dropout_rate, name="dropout_add")

    with tf.variable_scope(layer_name + "_conv"):
        upE = conv2d_layer(upB, filter_nbr, kernel_size, normalizer_fn=batchnorm,
                           activation_fn=leakyRelu,
                           normalizer_params={'is_training': training}, scope="conv1")
        upE = conv2d_layer(upE, filter_nbr, kernel_size, normalizer_fn=batchnorm,
                           activation_fn=leakyRelu,
                           normalizer_params={'is_training': training}, scope="conv2")
        upE = conv2d_layer(upE, filter_nbr, kernel_size, normalizer_fn=batchnorm,
                           activation_fn=leakyRelu,
                           normalizer_params={'is_training': training}, scope="conv3")
        upE = dropout_layer(upE, rate=dropout_rate, name="dropout_conv")

        print(str(layer_name) + str(upE.shape.as_list()))

        return upE


def upBlock1d(input_layer, skip_layer, filter_nbr, dropout_rate, kernel_size=(3, 3), layer_name="dec", training=True):
    with tf.variable_scope(layer_name + "_up"):
        upA = conv2d_trans_layer(input_layer, filter_nbr, kernel_size, 2, normalizer_fn=batchnorm,
                                 activation_fn=leakyRelu,
                                 normalizer_params={'is_training': training}, scope="tconv")
        upA = dropout_layer(upA, rate=dropout_rate, name="dropout")

    with tf.variable_scope(layer_name + "_add"):
        upB = tf.add(upA, skip_layer, name="add")
        upB = dropout_layer(upB, rate=dropout_rate, name="dropout_add")

    with tf.variable_scope(layer_name + "_conv"):
        upE = conv2d_layer(upB, filter_nbr, kernel_size, normalizer_fn=batchnorm,
                           activation_fn=leakyRelu,
                           normalizer_params={'is_training': training}, scope="conv1")
        upE = conv2d_layer(upE, filter_nbr, kernel_size, normalizer_fn=batchnorm,
                           activation_fn=leakyRelu,
                           normalizer_params={'is_training': training}, scope="conv2")
        upE = conv2d_layer(upE, filter_nbr, kernel_size, normalizer_fn=batchnorm,
                           activation_fn=leakyRelu,
                           normalizer_params={'is_training': training}, scope="conv3")
        upE = dropout_layer(upE, rate=dropout_rate, name="dropout_conv")

        print(str(layer_name) + str(upE.shape.as_list()))

        return upE


def create_SalsaNet(input_img, num_classes=3, dropout_rate=0.5, is_training=False, kernel_number=32):

    print("--------------- SalsaNet model --------------------")
    print("input", input_img.shape.as_list())

    down0c, down0b = resBlock(input_img, filter_nbr=kernel_number, dropout_rate=dropout_rate,
                              kernel_size=3, stride=1, layer_name="res0", training=is_training, repetition=1)
    down1c, down1b = resBlock(down0c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                              kernel_size=3, stride=1, layer_name="res1", training=is_training, repetition=1)
    down2c, down2b = resBlock(down1c, filter_nbr=4 * kernel_number, dropout_rate=dropout_rate,
                              kernel_size=3, stride=1, layer_name="res2", training=is_training, repetition=1)
    down3c, down3b = resBlock(down2c, filter_nbr=8 * kernel_number, dropout_rate=dropout_rate,
                              kernel_size=3, stride=1, layer_name="res3", training=is_training, repetition=1)
    down4b = resBlock(down3c, filter_nbr=8 * kernel_number, dropout_rate=dropout_rate, kernel_size=3,
                      stride=1, layer_name="res4", training=is_training, pooling=False, repetition=1)

    up3e = upBlock(down4b, down3b,  filter_nbr=8 * kernel_number, dropout_rate=dropout_rate,
                   kernel_size=(3, 3), layer_name="up3", training=is_training)
    up2e = upBlock(up3e, down2b,  filter_nbr=4 * kernel_number, dropout_rate=dropout_rate,
                   kernel_size=(3, 3), layer_name="up2", training=is_training)
    up1e = upBlock(up2e, down1b,  filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                   kernel_size=(3, 3), layer_name="up1", training=is_training)
    up0e = upBlock(up1e, down0b,  filter_nbr=kernel_number, dropout_rate=dropout_rate,
                   kernel_size=(3, 3), layer_name="up0", training=is_training)

    with tf.variable_scope('logits'):
        logits = conv2d_layer(up0e, num_classes, [1, 1], activation_fn=None)
        print("logits", logits.shape.as_list())

    return logits


def create_SalsaNet_laser(input_img, num_classes=3, dropout_rate=0.5, is_training=False, kernel_number=32):

    print("--------------- SalsaNet_laser model --------------------")
    print("input", input_img.shape.as_list())

    with tf.variable_scope("laser_block"):
        down0c, down0b = resBlock(input_img, filter_nbr=kernel_number, dropout_rate=dropout_rate,
                                  kernel_size=3, stride=1, layer_name="res0", training=is_training, repetition=1)
        down1c, down1b = resBlock(down0c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                                  kernel_size=3, stride=1, layer_name="res1", training=is_training, repetition=1)
        down2b = resBlock(down1c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate, kernel_size=3,
                          stride=1, layer_name="res2", training=is_training, pooling=False, repetition=1)

        up1e = upBlock(down2b, down1b,  filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                       kernel_size=(3, 3), layer_name="up1", training=is_training)
        up0e = upBlock(up1e, down0b,  filter_nbr=kernel_number, dropout_rate=dropout_rate,
                       kernel_size=(3, 3), layer_name="up0", training=is_training)

    with tf.variable_scope('logits'):
        logits = conv2d_layer(up0e, num_classes, [1, 1], activation_fn=None)
        print("logits", logits.shape.as_list())

    return logits

    print("--------------------------------------------------")


def create_SalsaNet_decoder_fusion(input_laser, input_depth, num_classes=3, dropout_rate=0.5, is_training=False, kernel_number=32):

    print("--------------- SalsaNet_decoder_fusion model --------------------")
    print("input_laser", input_laser.shape.as_list())
    print("input_depth", input_depth.shape.as_list())
    #laser_img_shape = (None, 60, 80)
    #depth_img_shape = (None, 240, 320)

    is_training_laser = True

    print("--- laser ---")
    with tf.variable_scope("laser_block"):
        # [ ,30, 40, 32]
        down0c, down0b = resBlock(input_laser, filter_nbr=kernel_number, dropout_rate=dropout_rate,
                                  kernel_size=3, stride=1, layer_name="res0", training=is_training_laser, repetition=1)
        # [ ,15, 20, 64]
        down1c, down1b = resBlock(down0c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                                  kernel_size=3, stride=1, layer_name="res1", training=is_training_laser, repetition=1)
        # [ ,15, 20, 64]
        down2b = resBlock(down1c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate, kernel_size=3,
                          stride=1, layer_name="res2", training=is_training_laser, pooling=False, repetition=1)
        # [ ,30, 40, 32]
        up1e = upBlock(down2b, down1b,  filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                       kernel_size=(3, 3), layer_name="up1", training=is_training_laser)
        # [ ,60, 80, 32]
        up0e = upBlock(up1e, down0b,  filter_nbr=kernel_number, dropout_rate=dropout_rate,
                       kernel_size=(3, 3), layer_name="up0", training=is_training_laser)
    print("--- depth ---")
    with tf.variable_scope("depth_block"):
        # [ ,120, 160, 32]
        ddown0c, ddown0b = resBlock(input_depth, filter_nbr=kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres0", training=is_training, repetition=1)
        # [ ,60, 80, 64]
        ddown1c, ddown1b = resBlock(ddown0c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres1", training=is_training, repetition=1)
        # [ ,30, 40, 128]
        ddown2c, ddown2b = resBlock(ddown1c, filter_nbr=4 * kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres2", training=is_training, repetition=1)
        # [ ,15, 20, 256]
        ddown3c, ddown3b = resBlock(ddown2c, filter_nbr=8 * kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres3", training=is_training, repetition=1)
        # [ ,15, 20, 256]
        ddown4b = resBlock(ddown3c, filter_nbr=8 * kernel_number, dropout_rate=dropout_rate, kernel_size=3,
                           stride=1, layer_name="dres4", training=is_training, pooling=False, repetition=1)

        # concatenation [ ,15, 20, 256+64]
        ddown4b = tf.concat(axis=3, values=[down2b, ddown4b])
        print("concat ddown4b", ddown4b.shape.as_list())
        
        '''
        # fusion 1
        ddown4_0c, ddown4_0b = resBlock(ddown4b, filter_nbr=16 * kernel_number, dropout_rate=dropout_rate,
                                        kernel_size=3, stride=1, layer_name="fusion_0a", training=is_training, repetition=1)
        ddown4_1c = resBlock(ddown4_0c, filter_nbr=16 * kernel_number, dropout_rate=dropout_rate,
                             kernel_size=3, stride=1, layer_name="fusion_1a", training=is_training, pooling=False, repetition=1)
        ddown4_2c = upBlock(ddown4_1c, ddown4_0b,  filter_nbr=16 * kernel_number, dropout_rate=dropout_rate,
                            kernel_size=(3, 3), layer_name="fusion_2a", training=is_training)
        exit()
        '''

        # [ ,30, 40, 256]
        dup3e = upBlock(ddown4b, ddown3b,  filter_nbr=8 * kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup3", training=is_training)

        # concatenation [ ,30, 40, 256+64]
        dup3e = tf.concat(axis=3, values=[up1e, dup3e])
        print("concat dup3", dup3e.shape.as_list())

        # [ ,60, 80, 128]
        dup2e = upBlock(dup3e, ddown2b,  filter_nbr=4 * kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup2", training=is_training)
        # concatenation [ ,60, 80, 128+32]
        dup2e = tf.concat(axis=3, values=[up0e, dup2e])
        print("concat dup2", dup2e.shape.as_list())
        # [ ,120, 160, 64]
        dup1e = upBlock(dup2e, ddown1b,  filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup1", training=is_training)
        # [ ,240, 320, 32]
        dup0e = upBlock(dup1e, ddown0b,  filter_nbr=kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup0", training=is_training)

    with tf.variable_scope('logits'):
        # [ ,240, 320, 5]
        logits = conv2d_layer(dup0e, num_classes, [1, 1], activation_fn=None)
        print("logits", logits.shape.as_list())

    return logits

    print("--------------------------------------------------")


def create_SalsaNet_encoder_fusion(input_laser, input_depth, num_classes=3, dropout_rate=0.5, is_training=False, kernel_number=32):

    print("--------------- SalsaNet_encoder_fusion model --------------------")
    print("input_laser", input_laser.shape.as_list())
    print("input_depth", input_depth.shape.as_list())
    #laser_img_shape = (None, 60, 80)
    #depth_img_shape = (None, 240, 320)

    print("--- laser ---")
    with tf.variable_scope("laser_block"):
        # [ ,30, 40, 32]
        down0c, down0b = resBlock(input_laser, filter_nbr=kernel_number, dropout_rate=dropout_rate,
                                  kernel_size=3, stride=1, layer_name="res0", training=False, repetition=1)
        # [ ,15, 20, 64]
        down1c, down1b = resBlock(down0c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                                  kernel_size=3, stride=1, layer_name="res1", training=False, repetition=1)
        # [ ,15, 20, 64]
        down2b = resBlock(down1c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate, kernel_size=3,
                          stride=1, layer_name="res2", training=False, pooling=False, repetition=1)
        # [ ,30, 40, 32]
        up1e = upBlock(down2b, down1b,  filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                       kernel_size=(3, 3), layer_name="up1", training=False)
        # [ ,60, 80, 32]
        up0e = upBlock(up1e, down0b,  filter_nbr=kernel_number, dropout_rate=dropout_rate,
                       kernel_size=(3, 3), layer_name="up0", training=False)
    print("--- depth ---")
    with tf.variable_scope("depth_block"):
        # [ ,120, 160, 32]
        ddown0c, ddown0b = resBlock(input_depth, filter_nbr=kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres0", training=is_training, repetition=1)
        # [ ,60, 80, 64]
        ddown1c, ddown1b = resBlock(ddown0c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres1", training=is_training, repetition=1)
        # [ ,30, 40, 128]
        ddown2c, ddown2b = resBlock(ddown1c, filter_nbr=4 * kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres2", training=is_training, repetition=1)
        # concatenation [ ,30, 40, 128+32]
        ddown2c = tf.concat(axis=3, values=[ddown2c, down0c])
        print("concat dres2", ddown2c.shape.as_list())
        # [ ,15, 20, 256]
        ddown3c, ddown3b = resBlock(ddown2c, filter_nbr=8 * kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres3", training=is_training, repetition=1)
        # concatenation [ ,15, 20, 256+64]
        ddown3c = tf.concat(axis=3, values=[ddown3c, down1c])
        print("concat dres3", ddown3c.shape.as_list())
        # [ ,15, 20, 256]
        ddown4b = resBlock(ddown3c, filter_nbr=8 * kernel_number, dropout_rate=dropout_rate, kernel_size=3,
                           stride=1, layer_name="dres4", training=is_training, pooling=False, repetition=1)
        # concatenation [ ,15, 20, 256+64]
        ddown4b = tf.concat(axis=3, values=[ddown4b, down2b])
        print("concat dres4", ddown4b.shape.as_list())
        # [ ,30, 40, 256]
        dup3e = upBlock(ddown4b, ddown3b,  filter_nbr=8 * kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup3", training=is_training)
        # [ ,60, 80, 128]
        dup2e = upBlock(dup3e, ddown2b,  filter_nbr=4 * kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup2", training=is_training)
        # [ ,120, 160, 64]
        dup1e = upBlock(dup2e, ddown1b,  filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup1", training=is_training)
        # [ ,240, 320, 32]
        dup0e = upBlock(dup1e, ddown0b,  filter_nbr=kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup0", training=is_training)

    with tf.variable_scope('logits'):
        # [ ,240, 320, 5]
        logits = conv2d_layer(dup0e, num_classes, [1, 1], activation_fn=None)
        print("logits", logits.shape.as_list())

    return logits

    print("--------------------------------------------------")


def create_SalsaNet_encoder_decoder_fusion(input_laser, input_depth, num_classes=3, dropout_rate=0.5, is_training=False, kernel_number=32):

    print("--------------- SalsaNet_encoder_fusion model --------------------")
    print("input_laser", input_laser.shape.as_list())
    print("input_depth", input_depth.shape.as_list())
    #laser_img_shape = (None, 60, 80)
    #depth_img_shape = (None, 240, 320)

    print("--- laser ---")
    with tf.variable_scope("laser_block"):
        # [ ,30, 40, 32]
        down0c, down0b = resBlock(input_laser, filter_nbr=kernel_number, dropout_rate=dropout_rate,
                                  kernel_size=3, stride=1, layer_name="res0", training=is_training, repetition=1)
        # [ ,15, 20, 64]
        down1c, down1b = resBlock(down0c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                                  kernel_size=3, stride=1, layer_name="res1", training=is_training, repetition=1)
        # [ ,15, 20, 64]
        down2b = resBlock(down1c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate, kernel_size=3,
                          stride=1, layer_name="res2", training=is_training, pooling=False, repetition=1)
        # [ ,30, 40, 32]
        up1e = upBlock(down2b, down1b,  filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                       kernel_size=(3, 3), layer_name="up1", training=is_training)
        # [ ,60, 80, 32]
        up0e = upBlock(up1e, down0b,  filter_nbr=kernel_number, dropout_rate=dropout_rate,
                       kernel_size=(3, 3), layer_name="up0", training=is_training)
    print("--- depth ---")
    with tf.variable_scope("depth_block"):
        # [ ,120, 160, 32]
        ddown0c, ddown0b = resBlock(input_depth, filter_nbr=kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres0", training=is_training, repetition=1)
        # [ ,60, 80, 64]
        ddown1c, ddown1b = resBlock(ddown0c, filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres1", training=is_training, repetition=1)
        # [ ,30, 40, 128]
        ddown2c, ddown2b = resBlock(ddown1c, filter_nbr=4 * kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres2", training=is_training, repetition=1)
        # concatenation [ ,30, 40, 128+32]
        ddown2c = tf.concat(axis=3, values=[ddown2c, down0c])
        print("concat dres2", ddown2c.shape.as_list())
        # [ ,15, 20, 256]
        ddown3c, ddown3b = resBlock(ddown2c, filter_nbr=8 * kernel_number, dropout_rate=dropout_rate,
                                    kernel_size=3, stride=1, layer_name="dres3", training=is_training, repetition=1)
        # concatenation [ ,15, 20, 256+64]
        ddown3c = tf.concat(axis=3, values=[ddown3c, down1c])
        print("concat dres3", ddown3c.shape.as_list())
        # [ ,15, 20, 256]
        ddown4b = resBlock(ddown3c, filter_nbr=8 * kernel_number, dropout_rate=dropout_rate, kernel_size=3,
                           stride=1, layer_name="dres4", training=is_training, pooling=False, repetition=1)
        # concatenation [ ,15, 20, 256+64]
        ddown4b = tf.concat(axis=3, values=[down2b, ddown4b])
        print("concat ddown4b", ddown4b.shape.as_list())
        # [ ,30, 40, 256]
        dup3e = upBlock(ddown4b, ddown3b,  filter_nbr=8 * kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup3", training=is_training)
        # concatenation [ ,30, 40, 256+64]
        dup3e = tf.concat(axis=3, values=[up1e, dup3e])
        print("concat dup3", dup3e.shape.as_list())
        # [ ,60, 80, 128]
        dup2e = upBlock(dup3e, ddown2b,  filter_nbr=4 * kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup2", training=is_training)
        # concatenation [ ,60, 80, 128+32]
        dup2e = tf.concat(axis=3, values=[up0e, dup2e])
        print("concat dup2", dup2e.shape.as_list())
        # [ ,120, 160, 64]
        dup1e = upBlock(dup2e, ddown1b,  filter_nbr=2 * kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup1", training=is_training)
        # [ ,240, 320, 32]
        dup0e = upBlock(dup1e, ddown0b,  filter_nbr=kernel_number, dropout_rate=dropout_rate,
                        kernel_size=(3, 3), layer_name="dup0", training=is_training)

    with tf.variable_scope('logits'):
        # [ ,240, 320, 5]
        logits = conv2d_layer(dup0e, num_classes, [1, 1], activation_fn=None)
        print("logits", logits.shape.as_list())

    return logits

    print("--------------------------------------------------")
