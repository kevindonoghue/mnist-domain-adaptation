import numpy as np
import os
import tarfile
import urllib.request
import torchvision
from skimage import io, transform

def get_computer_data():
    """
    Download and save dataset for labeled computer-generated data.
    """
    if os.path.exists('data/computer_images.npy') and  os.path.exists('data/computer_labels.npy'):
        return
    
    os.makedirs('data/', exist_ok=True)
    url = 'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz'

    ftpstream = urllib.request.urlopen(url)
    tar = tarfile.open(fileobj=ftpstream, mode='r|gz')
    tar.extractall()

    images = []
    labels = []
    for i in range(10):
        print(f'Processing Computer Digit: {i}')
        dirname = f'Sample00{i+1}' if i != 9 else f'Sample010'
        for filename in os.listdir(f'English/Fnt/{dirname}/'):
            path = f'English/Fnt/{dirname}/{filename}'
            if filename[-4:] == '.png':
                image = io.imread(path)
                image = transform.resize(image, (28, 28), cval=1).reshape(28*28)
                image = image.astype(np.uint8)
                images.append(image)
                labels.append(i)
            
    images = np.array(images)
    labels = np.array(labels)
    np.save('data/computer_images.npy', images)
    np.save('data/computer_labels.npy', labels)
    
def get_handwritten_data():
    """
    Download and save mnist dataset for handwritten digits.
    """
    paths = ('data/handwritten_train_images.npy',
             'data/handwritten_test_images.npy',
             'data/handwritten_train_labels.npy',
             'data/handwritten_test_labels.npy')
    if all([os.path.exists(path) for path in paths]):
        return
    
    os.makedirs('data/', exist_ok=True)
    mnist = dict()
    mnist['train'] = torchvision.datasets.MNIST('.', download=True)
    mnist['test'] = torchvision.datasets.MNIST('.', train=False, download=True)
    
    images = {'train': [], 'test': []}
    labels = {'train': [], 'test': []}
    
    for phase in ('train', 'test'):
        print(f'Processing mnist {phase} set')
        for image, label in mnist[phase]:
            image = np.array(image)
            image = 255 - image
            images[phase].append(image)
            labels[phase].append(label)
            
    images['train'] = np.array(images['train'])
    images['test'] = np.array(images['test'])
    labels['train'] = np.array(labels['train'])
    labels['test'] = np.array(labels['test'])
    np.save('data/handwritten_train_images.npy', images['train'])
    np.save('data/handwritten_test_images.npy', images['test'])
    np.save('data/handwritten_train_labels.npy', labels['train'])
    np.save('data/handwritten_test_labels.npy', labels['test'])
    
def get_data():
    get_computer_data()
    get_handwritten_data()
    
    
def smooth(a, n):
    """
    Helper function used in graphing progress in Net class.
    """
    a = np.array(a)
    k = len(a) // n
    b = np.array_split(a, k)
    c = np.concatenate([np.array([x.mean()]*len(x)) for x in b])
    return c