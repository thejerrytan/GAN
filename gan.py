from __future__ import print_function
import keras, os, tables
from keras.utils.io_utils import HDF5Matrix
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import LeakyReLU

class GAN():
    def __init__(self):
        self.img_rows =  512
        self.img_cols = 512
        self.channels = 3
        self.img_shape = (self.channels, self.img_rows, self.img_cols)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(200,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (200,)
        
        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.channels, self.img_rows, self.img_cols)
        
        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        # X_train = load_dataset()

        # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            start_idx = (epoch*half_batch)     % DATASET_SIZE
            end_idx   = ((epoch+1)*half_batch) % DATASET_SIZE
            if end_idx < start_idx: # wrap around, skip this iteration
                continue
            imgs, _, _, _ = load_data(None, start_idx, 0, half_batch, 0)

            noise = np.random.normal(0, 1, (half_batch, 200))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 200))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch, save_interval)

    def save_imgs(self, epoch, save_interval=30):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (1, 200))
        gen_imgs = self.generator.predict(noise)
        img_shape = (1, 512, 512, 3)
        gen_imgs = np.reshape(gen_imgs, img_shape)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        plt.imshow(gen_imgs[0, :, :, :])
        plt.savefig("images/%d.jpg" % (epoch / save_interval))
        plt.close()

def generate_dataset():
    import os, glob
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    FOLDER = "car_images/"
    FILENAME = "car_dataset.h5"
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    TARGET_SIZE = 50000
    os.chdir(ROOT)
    fd = tables.open_file(FILENAME, mode='w')
    atom = tables.Float64Atom()
    filters = tables.Filters(complevel=5, complib='blosc')
    dataset = fd.create_earray(f.root, 'data', atom, (0, 3, IMG_WIDTH, IMG_HEIGHT), filters=filters, expectedrows=TARGET_SIZE)
    
    os.chdir(ROOT + "/" + FOLDER)
    count = 0
    for f in glob.glob("*.jpg"):
        img = Image.open(f)
        count += 1
        print("%d : %s" % (count, f))
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        arr = np.asarray(img)
        arr = np.reshape(arr, (1, arr.shape[2], arr.shape[1], arr.shape[0]))
        dataset.append(arr)

    fd.close()

def normalize_data(X_train):
    return (X_train.astype(np.float32) - 127.5) / 127.5

def load_data(datapath, train_start, test_start, n_training_examples, n_test_examples):
    global DATASET_SIZE
    FILENAME = "D:\DeepLearning\car_dataset.h5"
    datapath = FILENAME if datapath is None else datapath
    os.chdir(ROOT)
    X_train = HDF5Matrix(datapath, 'data', train_start, train_start+n_training_examples, normalizer=normalize_data)
    y_train = HDF5Matrix(datapath, 'data', train_start, train_start+n_training_examples)
    X_test  = HDF5Matrix(datapath, 'data', test_start, test_start+n_test_examples, normalizer=normalize_data)
    y_test  = HDF5Matrix(datapath, 'data', test_start, test_start+n_test_examples)
    return X_train, y_train, X_test, y_test

def get_rgb_from_rgba_img(img_path):
    from PIL import Image
    png = Image.open(img_path).convert('RGBA')
    background = Image.new('RGBA', png.size, (255,255,255))
    alpha_composite = Image.alpha_composite(background, png)
    return alpha_composite

def generate_video():
    import glob, os, subprocess
    os.chdir("images")
    subprocess.call([
        'ffmpeg', '-start_number', '1', '-framerate', '10', '-i', '%d.jpg', '-r', '30', '-pix_fmt', 'yuv420p',
        'car_gan.mp4'
    ])
    # for file_name in glob.glob("*.png"):
    #    os.remove(file_name)

if __name__ == '__main__':
    ROOT = os.getcwd()
    DATASET_SIZE = 46751
    gan = GAN()
    gan.train(epochs=30000, batch_size=16, save_interval=30)
    generate_video()