import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import glob
import sys
import math

train_batch_size =64
valid_batch_size=64
img_size = 100
classes_names = []
dataset_file = glob.glob("dataset/asl_alphabet_train/*")


for f in dataset_file:
    classes_names.append(f[27:])  #  the name of the subfolders is the name of the class
num_classes = len(classes_names)


def parser(record):
    # a parsing function to parse the tfrecords
    keys_to_features = {
        "img_raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)

    }
    parsed = tf.parse_single_example(record,keys_to_features)  # parsing one example from the example buffer from the tfrecord using the keys
    image = tf.decode_raw(parsed["img_raw"], tf.float32)  # decoding ( bytes -> tf.float32)
    #image= tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[img_size, img_size, 3])  # reshaping images
    label = parsed["label"]  # casting labels to int32
    label = tf.one_hot(indices=label, depth=num_classes) # transform to one hot encoding
    # with tf.Session() as session:
    #     print(session.run(label))
    return image, label


def input_fn(filenames, batch_size, train_bool=True):
    # from tfrecord to iterable data
    dataset = tf.data.TFRecordDataset(filenames=filenames)#,num_parallel_reads=40)  # instantiantion of an object from class TFRecordDataset
    dataset = dataset.map(parser)  # maps a function to the dataset
    if train_bool:
        dataset = dataset.shuffle(buffer_size=2048)
        repeat = 1  # if in training mode allow reading data infinitely
    else:
        repeat = 1  # if in validation or test allow max 1 read
    dataset = dataset.repeat(repeat)
    dataset = dataset.batch(batch_size)  #  define bach size
    # iterator= dataset.make_one_shot_iterator()# making the iterator
    # images_batch, labels_batch=iterator.get_next()# getting the data
    # x= {'image': images_batch}
    # y= labels_batch

    return dataset  # x, y

def train_input_fn():
    return input_fn(filenames=["TFrecords/train.tfrecords"],batch_size=train_batch_size)

def val_input_fn():
    return input_fn(filenames=["TFrecords/val.tfrecords"],batch_size=valid_batch_size, train_bool=False)

def test_input_fn():
    return input_fn(filenames=["TFrecords/test.tfrecords"])

# def conv_layer_max2pool(Input, num_output_channels, conv_filter_size,conv_strides, pool_filter_size, pool_strides,POOL=True):
#     # a function to create convulional layers, parameters are :
#     #       num_output_channels : number of the output filters
#     #       conv_filters_size: size of the convolution filter it should be a 2-D tuple
#     #       conv-strides: strides of the convolution. It's assumes that the strides over the height are the same as over the width
#     #       pool_filter_strides: as the conv_filter_size but for the pooling filter
#     #       pool_strides: as the conv_strides but for the pooling
#
#     filter_shape= [conv_filter_size[0], conv_filter_size[1], Input.get_shape().as_list()[3], num_output_channels] #creating the shape of the filter to create the weights of the convolution
#     W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01)) #creating the weights
#     conv= tf.nn.conv2d(Input, W, [1,conv_strides,conv_strides,1], padding="SAME") # creating the convolutional layer
#
#     bias=tf.Variable(tf.zeros([num_output_channels])) # creating the biasis
#
#     conv=tf.nn.bias_add(conv, bias)
#     conv=tf.nn.relu(conv)
#
#     #max pooling
#     if POOL :
#         conv=tf.nn.max_pool(conv, [1, pool_filter_size[0], pool_filter_size[1], 1],[1, pool_strides, pool_strides, 1], padding="SAME")
#
#     return conv

# def model_fn(X, keep_prob):
#
#     conv1=conv_layer_max2pool(X,num_output_channels=96, conv_filter_size=(11,11), conv_strides=4, pool_filter_size=(3,3), pool_strides=2)
#
#     conv2=conv_layer_max2pool(conv1,num_output_channels=256, conv_filter_size=(5,5), conv_strides=1, pool_filter_size=(3,3), pool_strides=2)
#
#     conv3=conv_layer_max2pool(conv2,num_output_channels=384, conv_filter_size=(3,3), conv_strides=1, pool_filter_size=(2,2), pool_strides=2, POOL=False)
#     conv4=conv_layer_max2pool(conv3,num_output_channels=384, conv_filter_size=(3,3), conv_strides=1, pool_filter_size=(2,2), pool_strides=2, POOL=False)
#     conv5=conv_layer_max2pool(conv4,num_output_channels=256, conv_filter_size=(3,3), conv_strides=1, pool_filter_size=(3,3), pool_strides=2)
#
#
#
#
#     #conv2= tf.nn.dropout(conv5, keep_prob)
#
#     # conv3 = conv_layer_max2pool(conv2, num_output_channels=128, conv_filter_size=(3, 3), conv_strides=2,
#     #                             pool_filter_size=(2, 2), pool_strides=2)
#     #
#     # conv4 = conv_layer_max2pool(conv3, num_output_channels=128, conv_filter_size=(2, 2), conv_strides=2,
#     #                             pool_filter_size=(2, 2), pool_strides=2)
#
#
#
#     flat_layer= tf.layers.flatten(conv5)
#
#     #FC layers
#
#     dense1=tf.layers.dense(flat_layer, 4096, activation=tf.nn.relu)
#     dense1=tf.nn.dropout(dense1, keep_prob)
#
#     dense2=tf.layers.dense(dense1, 4096, activation=tf.nn.relu)
#     dense2=tf.nn.dropout(dense2, keep_prob)
#
#     dense3=tf.layers.dense(dense2, num_classes, activation=tf.nn.relu)
#     output=tf.nn.dropout(dense3, keep_prob)
#
#
#     #output layer
#     #output=tf.layers.dense(dense3, num_classes)
#     return output


#Remove previous weights, bias, inputs...
tf.reset_default_graph()

# place holdes for features, labels and keep_prob
x=tf.placeholder(tf.float32, [None, img_size, img_size, 3] , name="x")
y=tf.placeholder(tf.int64, [None, num_classes], name="y")
keep_prob=tf.placeholder(tf.float32, name="keep_prob")




 #Model

 #conv1
#conv1=conv_layer_max2pool(X,num_output_channels=96, conv_filter_size=(11,11), conv_strides=4, pool_filter_size=(3,3), pool_strides=2)

filter_shape= [11, 11, 3, 96] #creating the shape of the filter to create the weights of the convolution
W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01)) #creating the weights
conv= tf.nn.conv2d(x, W1, [1,4,4,1], padding="SAME") # creating the convolutional layer

bias1=tf.Variable(tf.zeros([96])) # creating the biasis

conv=tf.nn.bias_add(conv, bias1)
conv=tf.nn.relu(conv)
# conv=conv,[bias1],"waaaaaaaahya rbi moulay")

conv=tf.nn.max_pool(conv, [1, 3, 3, 1],[1, 2, 2, 1], padding="SAME")

#conv2
# conv2=conv_layer_max2pool(conv1,num_output_channels=256, conv_filter_size=(5,5), conv_strides=1, pool_filter_size=(3,3), pool_strides=2)
filter_shape= [5, 5, 96, 256] #creating the shape of the filter to create the weights of the convolution
W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01)) #creating the weights
conv= tf.nn.conv2d(conv, W2, [1,1,1,1], padding="SAME") # creating the convolutional layer

bias2=tf.Variable(tf.zeros([256])) # creating the biasis

conv=tf.nn.bias_add(conv, bias2)
conv=tf.nn.relu(conv)

conv=tf.nn.max_pool(conv, [1, 3, 3, 1],[1, 2, 2, 1], padding="SAME")

#conv3
# conv3=conv_layer_max2pool(conv2,num_output_channels=384, conv_filter_size=(3,3), conv_strides=1, pool_filter_size=(2,2), pool_strides=2, POOL=False)
filter_shape= [3, 3,256, 384] #creating the shape of the filter to create the weights of the convolution
W3 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01)) #creating the weights
conv= tf.nn.conv2d(conv, W3, [1,1,1,1], padding="SAME") # creating the convolutional layer

bias3=tf.Variable(tf.zeros([384])) # creating the biasis

conv=tf.nn.bias_add(conv, bias3)
conv=tf.nn.relu(conv)

#conv4
# conv4=conv_layer_max2pool(conv3,num_output_channels=384, conv_filter_size=(3,3), conv_strides=1, pool_filter_size=(2,2), pool_strides=2, POOL=False)
filter_shape= [3, 3, 384, 384] #creating the shape of the filter to create the weights of the convolution
W4 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01)) #creating the weights
conv= tf.nn.conv2d(conv, W4, [1,1,1,1], padding="SAME") # creating the convolutional layer

bias4=tf.Variable(tf.zeros([384])) # creating the biasis


conv=tf.nn.bias_add(conv, bias4)
conv=tf.nn.relu(conv)
# conv=tf.Print(conv,[W4],"waaaaaaaahya rbi moulay")
#conv5
# conv5=conv_layer_max2pool(conv4,num_output_channels=256, conv_filter_size=(3,3), conv_strides=1, pool_filter_size=(3,3), pool_strides=2)
filter_shape= [3, 3, 384, 256] #creating the shape of the filter to create the weights of the convolution
W5 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01)) #creating the weights
conv= tf.nn.conv2d(conv, W5, [1,1,1,1], padding="SAME") # creating the convolutional layer

bias5=tf.Variable(tf.zeros([256])) # creating the biasis

conv=tf.nn.bias_add(conv, bias5)
conv=tf.nn.relu(conv)
conv=tf.nn.max_pool(conv, [1, 3, 3, 1],[1, 2, 2, 1], padding="SAME")



flat_layer= tf.layers.flatten(conv)

#FC layers

dense1=tf.layers.dense(flat_layer, 4096, activation=tf.nn.relu)
dense1=tf.nn.dropout(dense1, keep_prob)

dense2=tf.layers.dense(dense1, 4096, activation=tf.nn.relu)
dense2=tf.nn.dropout(dense2, keep_prob)

dense3=tf.layers.dense(dense2, num_classes)#, activation=tf.nn.relu)
# output=tf.nn.dropout(dense3, keep_prob)


#output layer
#output=tf.layers.dense(dense3, num_classes)


#logits
logits= tf.identity(dense3 , name="logits")

# loss=
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels=y, logits=logits))

#optimizer
optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

#Accuracy
# y_max=tf.Print(tf.argmax(y,1),[tf.shape(y),y, tf.argmax(y,1)],"y",summarize=30)
# logits_max=tf.Print(tf.argmax(logits,1),[tf.shape(logits),logits,tf.argmax(logits,1)],"logits",summarize=30)
pred=tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(pred,tf.float32), name="accuracy")


train_dataset= train_input_fn()
train_iterator=train_dataset.make_initializable_iterator()
features, labels= train_iterator.get_next()

valid_dataset=val_input_fn()
valid_iterator=valid_dataset.make_initializable_iterator()
valid_features, valid_labels=valid_iterator.get_next()


save_model_path="Model/model0/"

epochs= 10
keep_probability=0.5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #num_images=train_input_fn().get_shape().as_list[0]
    #training_cycle
    for epoch in range(epochs):
        #for _ in range(math.ceil(num_images/batch_size)):
        sess.run(train_iterator.initializer)
        sess.run(valid_iterator.initializer)
        count=0
        while True  :
            try:
                count += 1
                img_batch, label_batch= sess.run([features,labels])
                #print([classes_names[i] for i in label_batch])
                #input()
                # for i,test in enumerate(img_batch):
                #     #print(label_batch)#classes_names[label_batch[1]])
                #     test2 = Image.fromarray(img_batch[i].astype('uint8'), 'RGB')
                #     test2.show(test2)
                #     input()
                # print("next batch")
                sess.run(optimizer, feed_dict={x: img_batch, y: label_batch, keep_prob: keep_probability})
                #print("batch number :", count)
                #print(count)
                l = sess.run(loss, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                #print("The loss is : {0}".format(l))
            except tf.errors.OutOfRangeError:
                print('Epoch {:>2}: '.format(epoch + 1), end='')
                l = sess.run(loss, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                break


        count=0
        valid_accuracy=0

        while True :
            try:
                valid_img_batch, valid_label_batch = sess.run([valid_features, valid_labels])
                valid_accuracy+=sess.run(accuracy, feed_dict={x: valid_img_batch, y:valid_label_batch, keep_prob:1.0})
                # test2=np.reshape(valid_img_batch,(img_size,img_size,3))
                # test2 = Image.fromarray(test2.astype('uint8'), 'RGB')
                # test2.show(test2)
                # print("ignoooooore : {0}".format(np.argmax(valid_label_batch,1)))
                # print(classes_names[int(input())])
                # input()
                #print("The loss is : {0}, and the Validation Accuracy is: {1}".format(l, valid_accuracy))
            except  tf.errors.OutOfRangeError:
                break
            count += 1
        valid_accuracy=valid_accuracy/count
        print("The loss is : {0}, and the Validation Accuracy is: {1}".format(l, valid_accuracy))

    saver=tf.train.Saver()
    saver_path=saver.save(sess,save_model_path)

    # Test_path="/home/hch/DL/Tensorflow_mini_projects/ASL-alphabet/dataset/asl_alphabet_test/nothing_test.jpg"
    # img = cv2.imread(Test_path)
    # img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    # img = img.astype(np.float32)  # /255
    # test2 = Image.fromarray(img.astype('uint8'), 'RGB')
    # img=np.reshape(img,(1,img_size, img_size,3))
    #
    # pred=sess.run(tf.argmax(logits,1), feed_dict={x:img, keep_prob:1.0})
    # #print(pred)
    # pred=classes_names[pred[0]]
    # test2.show(test2)
    # print(pred)
    # input()
