import tensorflow as tf
import cv2
import numpy as np
import glob

model_path="./Model/model0/"
loaded_graph=tf.Graph()
img_path="./dataset/asl_alphabet_test/space_test.jpg"
img_size=100

classes_names = np.load('class_names.npy')
# dataset_file = glob.glob("dataset/asl_alphabet_train/*")
# for f in dataset_file:
#     classes_names.append(f[27:])  #  the name of the subfolders is the name of the class
# num_classes = len(classes_names)
# print(np.array(classes_names,dtype=object))
# np.save('class_names',classes_names)


def get_img():
    img=cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)  # /255
    return img



def infer():
    with tf.Session(graph=loaded_graph) as sess:
        #load model
        loader=tf.train.import_meta_graph(model_path+'model.ckpt.meta')
        loader.restore(sess, "./Model/model0/model.ckpt")

        #get tensors
        loaded_x=loaded_graph.get_tensor_by_name('x:0')
        loaded_y=loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob=loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits=loaded_graph.get_tensor_by_name('logits/BiasAdd:0')

        img=get_img()
        img=np.transpose(np.reshape(img,(1,img_size,img_size,3)),(0,3,1,2))
        pred=sess.run(tf.argmax(loaded_logits,1),feed_dict={loaded_x:img,loaded_keep_prob:1.0})

        return pred


print(classes_names[infer()[0]])
input()



