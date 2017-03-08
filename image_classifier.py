# 500px ML Challenge by Bryan Lou

# What I have learned while conpleting the challenge:
# 1. Some the front end of TensorFlow.
# 2. How Tensorflow works (computational graphs, sessions, and some other stuff that is happening in the background)
# 3. Taking derivatives is very cheap from colah's blog. (regarding backpropagation in neural nets and computational graphs)
# 4. Reviewed softmax classification.
# 5. Adversarial images and how models can be "tricked" with them. 

# Notes:
# 1. Neural network was not used as it is time consuming to train on my laptop.
# 2. Results of the training could be saved to avoid retraining every time the code is being run.
# 3. There were very few commits made as I was invloved into process.
# 4. My usual branching model: master <=> dev <=> {feature names}
# 5. Tried to achieve high cohession and low coupling in my code.

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mimage
from matplotlib.backends.backend_pdf import PdfPages
import logging

SAMPLE_DATASET = input_data.read_data_sets('MNIST_data', one_hot=True)
# in px
SAMPLE_DIM = 28
SAMPLE_CLASSES = 10

SAMPLE_VAL_IMAGE = 0
SAMPLE_VAL_Y = 1

class ImageClassifier():
    def __init__(self, data_set, dimensions, classes):
        self.data_set = data_set
        self.dimensions = dimensions
        self.classes = classes
        self.sess = tf.InteractiveSession()

        # Tensor Flow Tutorial code
        self.x = tf.placeholder(tf.float32, shape=[None, dimensions*dimensions])
        self.y_ = tf.placeholder(tf.float32, shape=[None, classes])

        self.W = tf.Variable(tf.zeros([dimensions*dimensions, classes]))
        self.b = tf.Variable(tf.zeros([classes]))

        self.sess.run(tf.global_variables_initializer())

        self.y = tf.matmul(self.x,self.W) + self.b

        # With lower regularization it is easier to trick the model
        # as mentioned in the article, hence it was not introduced.
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    def train(self):
        # 1. Feed our model
        # 2. Collect samples of 2
        for i in range(1000):
            batch_xs, batch_ys = self.data_set.train.next_batch(100)
            self.train_step.run(feed_dict={self.x: batch_xs, self.y_: batch_ys})

    def get_samples_of(self, index):
        # Collect up to 10 samples
        result = []
        batch_xs, batch_ys = self.data_set.train.images, self.data_set.train.labels 
        for j in xrange(len(batch_xs)):
            sample = (batch_xs[j], batch_ys[j])

            if sample[SAMPLE_VAL_Y][index] and len(result) < 10:
                result += [sample]

        return result

    def classify_image(self, image):
        feed_dict = {self.x: [image]}
        return tf.argmax(self.sess.run(self.y, feed_dict), 1).eval()

    def reshape_image(self, image_1d):
        return tf.reshape(image_1d, [self.dimensions, self.dimensions])

    def get_weights_of(self, index):
        w_vals = self.sess.run(self.W)
        return [row[index] for row in w_vals]

class Interface():
    def plot_images(self, images):
        rows = len(images)
        cols = len(images[0])
        gs = gridspec.GridSpec(rows, cols, top=1., bottom=0., right=1., left=0., hspace=0.,
                    wspace=0.)

        for i, g in enumerate(gs):
            ax = plt.subplot(g)
            ax.imshow(images[i/cols][i%cols])
            ax.set_xticks([])
            ax.set_yticks([])
                
        plt.savefig("adv_result.png")
        plt.show()

    # for personal testing
    def show_image(self, image_1d):
        image = reshape_image(image_1d)
        plt.imshow(image.eval())

        plt.savefig("test.png")
        plt.show()

def get_adv_image(sample, delta):
    # using array instead of a tuple to keep it mutable
    return [sample, delta, sample + delta]

def reshape_adv_images(adv_images):
    for i, adv_image in enumerate(adv_images):
        for j, item in enumerate(adv_image):
            adv_images[i][j] = classifier.reshape_image(item).eval()

if __name__ == '__main__':
    classifier = ImageClassifier(SAMPLE_DATASET, SAMPLE_DIM, SAMPLE_CLASSES)
    classifier.train()
    # Create adversarial images using weights of "six" as our delta 
    adv_images = [get_adv_image(sample[SAMPLE_VAL_IMAGE], classifier.get_weights_of(6)) for sample in classifier.get_samples_of(2)]
    # Log classification results 
    # (Some adversarial images might not "trick" the model)
    results = [classifier.classify_image(adv_image[2]) for adv_image in adv_images]
    logging.info('classified as {}'.format(results))

    # reshape images to display in two dimensions before plotting
    reshape_adv_images(adv_images)

    # print adv_images
    interface = Interface()
    interface.plot_images(adv_images)


# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



