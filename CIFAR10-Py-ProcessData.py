import os
import numpy as np

base_dir = "./cifar-10-batches-py"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def pickle(file, obj):
    import pickle
    with open(file, 'wb') as file:
        pickle.dump(obj, file)

def covert_rgb(data):
    columns = np.zeros((32, 32, 3))
    for y in range(0, 32):
        rows = np.zeros((32, 3))
        for z in range(0, 32):
            r = data[y * 31 + z]
            g = data[(y * 31 + 1023) + z]
            b = data[(y * 31 + 2047) + z]
            rows[z] = [r, g, b]
        columns[y] = rows
    return columns

train_data = np.zeros((50000, 32, 32, 3))
train_labels = np.zeros((50000,))

test_data = np.zeros((10000, 32, 32, 3))
test_labels = np.zeros((10000,))

# process train data
for i in range(1, 6):
    dict = unpickle(os.path.join(base_dir, "data_batch_{}".format(i)))
    dict_data = dict['data']
    for x, data in enumerate(dict_data):
        columns = covert_rgb(data)
        train_data[((i - 1) * 10000) + x] = columns
        print(((i - 1) * 10000) + x)

    dict_labels = dict['labels']
    for x, label in enumerate(dict_labels):
        train_labels[((i - 1) * 10000) + x] = label

pickle(os.path.join(base_dir, 'train_data'), train_data)
pickle(os.path.join(base_dir, 'train_labels'), train_labels)

#process test data
test_dict = unpickle(os.path.join(base_dir, "test_batch"))

test_temp_data = test_dict['data']
for x, data in enumerate(test_temp_data):
    columns = covert_rgb(data)
    test_data[x] = columns
    print(x)

test_labels = test_dict['labels']

pickle(os.path.join(base_dir, 'test_data'), test_data)
pickle(os.path.join(base_dir, 'test_labels'), test_labels)

