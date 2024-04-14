import numpy as np
import matplotlib.pyplot as plt
import mnist
import logging
from time import time

class SGDMnistClassifier:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.data = list(zip(xs, ys))
        self.w = np.zeros((10, xs.shape[1]))

    def zero_one_loss(self, xs: np.ndarray, ys: np.ndarray):
        """
        Calculate the zero-one loss of the classifier on the given data.
        """
        correct = 0
        for x, y in list(zip(xs, ys)):
            correct += self.classify(x) == y
        return 1 - correct / len(xs)

    def shuffle(self):
        """
        Shuffles the data in place.
        """
        if self.data is None:
            raise ValueError("No data to shuffle.")
        np.random.shuffle(self.data)

    def p(self, x: np.ndarray, y: np.uint8) -> np.float64:
        """
        Calculate the probability of x being in class y.
        """
        return np.exp(self.w[y] @ x) / np.sum(np.exp(self.w @ x))

    def classify(self, x: np.ndarray):
        """
        Return the class of x with the highest probability.
        """
        return np.argmax([np.dot(x, self.w[y]) for y in range(10)])

    def loss(self, x: np.ndarray, y: np.uint8):
        """
        Returns the log loss
        """
        return -self.w[y] @ x + np.logaddexp.reduce(self.w @ x)

    def dloss(self, x: np.ndarray, y: np.uint8, y_true: np.uint8):
        """
        Returns the gradient of the loss function with respect to the weights.
        """
        if y != y_true:
            return x * (np.exp(-self.loss(x, y)))
        return x * (np.exp(-self.loss(x, y)) - 1)

    def learn(self, step_size, epochs):
        """
        Train the classifier using stochastic gradient descent.
        """
        if self.data is None:
            raise ValueError("No data to learn from.")

        losses = np.empty(epochs)

        for e in range(epochs):
            loss_values = []
            self.shuffle()
            logging.info(f"Epoch {e + 1} of {epochs}...")
            for x_i, y_i in self.data:
                loss_values.append(self.loss(x_i, y_i))
                for y_j in range(10):
                    self.w[y_j] -= step_size * self.dloss(x_i, np.uint8(y_j), y_i)

            losses[e] = np.mean(loss_values)

        return losses


def part_a():
    images = mnist.load_images("data/train-images-idx3-ubyte")
    reshaped_images = np.array([np.reshape(im, (784,)) for im in images])
    mean_image = np.reshape(np.mean(reshaped_images, axis=0), (28, 28))
    _, ax = plt.subplots()
    ax.imshow(mean_image)
    plt.show(block=False)


def part_c():
    # Load training data
    training_images = mnist.load_images("data/train-images-idx3-ubyte")
    training_images = np.array([np.reshape(im, (784,)) / 255 for im in training_images])
    training_labels = mnist.load_labels("data/train-labels-idx1-ubyte")

    # Load testing data
    testing_images = mnist.load_images("data/t10k-images-idx3-ubyte")
    testing_images = np.array([np.reshape(im, (784,)) / 255 for im in testing_images])
    testing_labels = mnist.load_labels("data/t10k-labels-idx1-ubyte")

    if TEST_LEARNING_RATES:
        start_time = time()


        # Initialise data
        step_sizes = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
        ]
        log_losses_epoch = np.empty(len(step_sizes), dtype=list)
        log_losses = np.empty(len(step_sizes), dtype=np.float64)
        zero_one_losses = np.empty(len(step_sizes), dtype=np.float64)

        # Collect data
        for i, step_size in enumerate(step_sizes):
            epoch_time = time()
            classifier = SGDMnistClassifier(training_images, training_labels)
            logging.info(
                f"Learning with step size {step_size} ({i + 1}/{len(step_sizes)})..."
            )
            log_losses_epoch[i] = classifier.learn(step_size, epochs=10)
            log_losses[i] = log_losses_epoch[i][-1]
            zero_one_losses[i] = classifier.zero_one_loss(testing_images, testing_labels)
            logging.info(f"Done in {np.round(time() - epoch_time, decimals=2)} seconds")

        logging.info(f"Total time: {np.round(time() - start_time, decimals=2)} seconds")

        # Plot data
        _, ax = plt.subplots()
        for i, step_size in enumerate(step_sizes):
            ax.plot(log_losses_epoch[i], label="Step size: " + str(step_size), marker="o")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log loss")
        ax.set_title("Log losses for different epochs and step sizes")

        _, ax = plt.subplots()
        for i, step_size in enumerate(step_sizes):
            ax.scatter(step_sizes, log_losses)
        ax.set_xlabel("Step size")
        ax.set_title("Final epoch log loss for different step sizes")
        ax.set_ylabel("Final epoch log loss")

        _, ax = plt.subplots()
        for i, step_size in enumerate(step_sizes):
            ax.scatter(step_sizes, zero_one_losses)
        ax.set_xlabel("Step size")
        ax.set_ylabel("Zero-one loss")
        ax.set_title("Zero-one loss for different step sizes")

    # Train with best step size
    step_size = 0.005
    logging.info(f"Learning with step size {step_size} (final)...")
    classifier = SGDMnistClassifier(training_images, training_labels)
    classifier.learn(0.01, epochs=10)

    # Test classifier
    zero_one_loss = classifier.zero_one_loss(training_images, training_labels)
    logging.info(f"Zero-one loss on test set: {zero_one_loss}")

TEST_LEARNING_RATES = False
def main():
    np.seterr(all="raise")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    part_a()
    part_c()
    plt.show()


if __name__ == "__main__":
    main()
