import numpy as np
import matplotlib.pyplot as plt

class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors = 3, metric = 'l2'):
        dists = None
        if metric == "l2":
            dists = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis = 1))
        else:
            dists = np.sum(np.absolute(self.train_images - test_image), axis = 1)

        nearest_indices = np.argpartition(dists, num_neighbors)[:num_neighbors]
        nearest_labels = self.train_labels[nearest_indices]
        most_common = np.bincount(nearest_labels.astype('int')).argmax()
        return most_common


train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', dtype='float')

test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', dtype='float')

def runModel(num_neighbors, metric):
    knn = KnnClassifier(train_images, train_labels)

    correct = 0
    total = len(test_images)
    predictions = np.array([knn.classify_image(image, num_neighbors, metric) for image in test_images])

    for i in range(total):
        predicted_label = predictions[i]
        if predicted_label == test_labels[i]:
            correct += 1

    if num_neighbors == 3:
        np.savetxt("results/predictii_3nn_l2_mnist.txt", predictions)

    return correct / total * 100

def plot_accuracy(accuracys_l1, accuracys_l2, num_neighbors):
    plt.figure(figsize=(10, 6))
    plt.plot(num_neighbors, accuracys_l1, 'o-', label='L1 Distance (Manhattan)')
    plt.plot(num_neighbors, accuracys_l2, 's-', label='L2 Distance (Euclidean)')

    plt.title('KNN Accuracy vs. Number of Neighbors')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy (%)')
    plt.xticks(num_neighbors)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def main():
    accuracys_l1 = []
    accuracys_l2 = []

    num_neighbors = [1, 3, 5, 7, 9]
    for num_neighbor in num_neighbors:
        accuracy_l1 = runModel(num_neighbor, 'l1')
        accuracy_l2 = runModel(num_neighbor, 'l2')

        accuracys_l1.append(accuracy_l1)
        accuracys_l2.append(accuracy_l2)
        print("Accuracy = {:.2f}% for num_neighbors = {} with dist={}".format(accuracy_l1, num_neighbor, 'l1'))
        print("Accuracy = {:.2f}% for num_neighbors = {} with dist={}".format(accuracy_l2, num_neighbor, 'l2'))

        print()

    np.savetxt("results/acuratete_l2.txt", accuracys_l2)

    plot_accuracy(accuracys_l1, accuracys_l2, num_neighbors)


if __name__ == "__main__":
    main()
