import glob
import numpy as np
from random import shuffle
import cv2
from tqdm import tqdm
import tensorflow as tf

#dataset file
dataset_file=glob.glob("dataset/asl_alphabet_train/*")


features_add=[]
labels=[]
classes_names=[]
img_size=100

#getting the addess of each image and its label and the name of each class
#the label will be the same as the index of the corresponding classe name : classes_names(label[i]) is the name of the classes of the image i 
for i,f in enumerate(dataset_file):
    classes_names.append(f[27:]) # the name of the subfolders is the name of the class
    subfolder_paths=glob.glob(f+"/*")
    for j,sub_f in enumerate(subfolder_paths):
        labels.append(i) # the label is the same for all the images in the subfolder
        features_add.append(sub_f)

features_add=np.asarray(features_add)
labels=np.asarray(labels)
classes_names=np.asarray(classes_names)

# shuffle the data 
tmp=list(zip(features_add, labels))
shuffle(tmp)
features_add, labels=zip(*tmp)

#Divide the data into 70% train, 20% validation and 10% test
train_add=features_add[0:int(0.7*len(features_add))]
train_labels=labels[0:int(0.7*len(features_add))]

val_add=features_add[int(0.7*len(features_add)):int(0.9*len(features_add))]
val_labels=labels[int(0.7*len(features_add)):int(0.9*len(features_add))]


test_add=features_add[int(0.9*len(features_add)):]
test_labels=labels[int(0.9*len(features_add)):]

def read_image(add):
    #read an image
    #no need to resize, all the images in this dataset are 200x200
    #cv2 images in BGR, it doesn't really matter for our netword
    img= cv2.imread(add)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img=img.astype(np.float32)#/255
    #img = np.asarray(img)
    return img


def _int64_feature(value):
    #convert value to int64 using tf.train.Int64List and creature a feature using tf.train.Feature
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    #convert value to bytes using tf.train.BytesList and creature a feature using tf.train.Feature
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def TFrecord_write(features_add,labels, name):
    # data->FeatureSet->Example protocol-> Serialized Example -> tfRecord
    with tf.python_io.TFRecordWriter("TFrecords/"+name+".tfrecords") as writer: # the TFwriter writer
        print("writing the "+name+" TFrecord")
        for i,add in enumerate(tqdm(features_add)):
            img_raw=read_image(add)
            height=img_raw.shape[0]
            width=img_raw.shape[1]
            depth=img_raw.shape[2]
            img_raw=img_raw.tostring()# convert each image to bytes
            example= tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': _int64_feature(int(labels[i])),
                        'img_raw': _bytes_feature(tf.compat.as_bytes(img_raw))
                    }))
            writer.write(example.SerializeToString())

#write the record
TFrecord_write(train_add,train_labels,"train")
TFrecord_write(val_add,val_labels,"val")
TFrecord_write(test_add,test_labels,"test")
