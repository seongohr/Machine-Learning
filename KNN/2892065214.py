# Name: Seongoh Ryoo

import numpy as np
from sklearn.decomposition import PCA
import sys

IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_IMAGES = 60000
NUM_INPUT = 800
IMAGE_FILE = "/train-images-idx3-ubyte"
LABEL_FILE = "/train-labels-idx1-ubyte"
OUTPUT_PATH = "./2892065214.txt"


def read_images(path):
    with open(path, 'rb') as f:
        f.read(16)
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        images = data.reshape(NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE)

        return images[:NUM_INPUT]


def read_labels(path):
    with open(path, 'rb') as f:
        # magic_number = int.from_bytes(f.read(4), 'big')
        # label_count = int.from_bytes(f.read(4), 'big')

        f.read(8)
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))

        return labels[:NUM_INPUT]


def write_file(predicted_labels, true_labels, path):
    result = ""
    for i in range(len(predicted_labels)):
        result += str(predicted_labels[i]) + " " + str(true_labels[i]) + "\n"

    with open(path, 'w') as f:
        f.write(result)


def split_train_test(images, labels, n):
    test_images = images[:n]
    training_images = images[n:]
    test_labels = labels[:n]
    training_labels = labels[n:]

    return test_images, training_images, test_labels, training_labels


def get_PCA_transformation(train, test, D):
    train_samples, train_r, train_c = train.shape
    train_data = train.reshape((train_samples, train_r*train_c))

    test_samples, test_r, test_c = test.shape
    test_data = test.reshape((test_samples, test_r*test_c))

    # PCA transform
    pca = PCA(n_components=D, svd_solver='full')
    pca.fit(train_data)

    transformed_train_data = pca.transform(train_data)
    transformed_test_data = pca.transform(test_data)

    return transformed_train_data, transformed_test_data


def get_euclidean_distance(train, test):
    result = []

    for t in test:
        result.append(np.sqrt(np.sum((t - train) ** 2, axis=1)))

    return result


def get_prediction(indices, distances, y_train):
    num_data = len(indices)  # number of test data
    num_k = len(indices[0])  # number of neighbors k
    labels = [[0 for i in range(NUM_LABELS)] for j in range(num_data)]
    predictions = []

    for i in range(num_data):
        for j in range(num_k):
            y = y_train[indices[i][j]]  # label
            labels[i][y] += 1/distances[i][j]  # weighted by inverse distance
        predicted_labels = labels[i].index(max(labels[i]))  # maximum scored index(0~9) -> label
        predictions.append(predicted_labels)

    return predictions


def knn(x_train, x_test, y_train, k):
    # distances from each test data to all the train data
    distances = get_euclidean_distance(x_train, x_test)

    # indexes of distance array sorted by distance in ascending order
    # label = y_train[index]
    indices = np.argsort(distances)

    # distance sorted in ascending order
    distances = np.sort(distances)

    # predicted labels of test data
    prediction = get_prediction(indices[:, :k], distances[:, :k], y_train)

    return prediction


def main():
    # parse the argument from command line
    arguments = sys.argv
    k = int(arguments[1])
    d = int(arguments[2])
    n = int(arguments[3])
    path = arguments[4]

    # read files
    images = read_images(path + IMAGE_FILE)
    labels = read_labels(path + LABEL_FILE)

    # split test and train data
    test_images, train_images, test_labels, train_labels = split_train_test(images, labels, n)
    # print(test_images[0].mean())

    # pca transform
    pca_train_images, pca_test_images = get_PCA_transformation(train_images, test_images, d)
    # print(pca_test_images[0])

    # KNN prediction
    predictions = knn(pca_train_images, pca_test_images, train_labels, k)

    # write a file
    write_file(predictions, test_labels, OUTPUT_PATH)


if __name__ == "__main__":
    main()