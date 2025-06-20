import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', dtype='float')

test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', dtype='float')

def values_to_bins(vals, bins):
    return np.digitize(vals, bins=bins[1:-1])

def run_model(num_bins):
    bins = np.linspace(start=0, stop=255, num=num_bins)

    train_disc = values_to_bins(train_images, bins)
    test_disc = values_to_bins(test_images, bins)

    model = MultinomialNB()
    model.fit(train_disc, train_labels)

    predictions = model.predict(test_disc)

    accuracy = model.score(test_disc, test_labels)
    return accuracy, predictions

def find_optimal_bins(bins_steps):
    best_accuracy = 0
    best_num_bins = bins_steps[0]

    for num_bins in bins_steps:
        accuracy, _ = run_model(num_bins)
        print("Accuracy = {:.2f}% with num_bins = {}".format(accuracy * 100, num_bins))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_bins = num_bins

    print("\nOptimal num_bins = {} with accuracy = {:.2f}%".format(best_num_bins, best_accuracy * 100))
    return best_num_bins

def show_misses(num_bins, num_imgs):
    _, predictions = run_model(num_bins)

    misses = np.where(predictions != test_labels)[0]

    num_to_show = min(num_imgs, len(misses))
    plt.figure(figsize=(28, 28))

    for i, index in enumerate(misses[:num_to_show]):
        plt.subplot(2, 5, i+1)
        plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
        plt.title(f"True: {int(test_labels[index])}, Pred: {int(predictions[index])}")
        plt.axis('off')

    plt.show()

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(classes)

    mat = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        mat[int(true_label), int(pred_label)] += 1

    return mat

def main():
    bins_steps = [3, 5, 7, 9, 11]

    optimal_bins = find_optimal_bins(bins_steps)
    show_misses(optimal_bins, 10)

    _, predictions = run_model(optimal_bins)
    conf_mat = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:")
    print(conf_mat)

if __name__ == "__main__":
    main()
