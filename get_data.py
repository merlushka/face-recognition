import sys
import os
import pickle
from scipy.misc import imread
from urllib import urlretrieve
if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve
import numpy as np
import tarfile

def download(filename, source='https://github.com/merlushka/face-recognition/releases/download/v1.0/'):
    print("Downloading %s" % filename)
    urlretrieve(source+filename, filename)
def unpack(filename):
    tar = tarfile.open(filename)
    tar.extractall()
    tar.close()

def load_dataset(task):  
    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    assert task in ('video', 'images'), 'Wrong task! The argument should be \'video\' or \'images\'.'
    def get_archived_data(filename, is_video=False):
        if not os.path.exists(filename):
            download(filename)
            unpack(filename)
        if is_video:
            data = {}
            for video in sorted(os.listdir(filename.split('.')[0])):
                data[int(video)]=[]
                for frame in os.listdir(filename.split('.')[0]+'/'+video):
                    data[int(video)].append(imread(filename.split('.')[0]+'/'+video+'/'+frame))
        else:
            data = []
            for image in sorted(os.listdir(filename.split('.')[0]), key = lambda x: int(x.split('.')[0])):
                data.append(imread(filename.split('.')[0]+'/'+image))

        return data

    if task == 'images':
        if not os.path.exists('y_train.pickle'):
            download('y_train.pickle')
        if not os.path.exists('y_test.pickle'):
            download('y_test.pickle')
        y_test = pickle.load(open('y_test.pickle', 'r'))
        y_train = pickle.load(open('y_train.pickle', 'r'))

        x_train = get_archived_data('train_1.tar.gz')
        x_test = get_archived_data('test_1.tar.gz')


    elif task == 'video':
        if not os.path.exists('train_video_labels.pickle'):
            download('train_video_labels.pickle')
        if not os.path.exists('video_labels.pickle'):
            download('video_labels.pickle') 
        y_test = pickle.load(open('video_labels.pickle', 'r'))
        y_train = pickle.load(open('train_video_labels.pickle', 'r'))

        x_train = get_archived_data('train_videos.tar.gz')
        x_test = get_archived_data('test_videos.tar.gz', is_video=True)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return x_train, y_train, x_test, y_test


def load_faces():
    if not os.path.exists('celeb_pics.pickle'):
        download('celeb_pics.pickle')
    with open ('celeb_pics.pickle', 'r') as f:
        people_faces = pickle.load(f)
    return people_faces