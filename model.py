import cv2
import numpy as np
import os
import csv
import sklearn
from sklearn.model_selection import train_test_split



DST_IMG_ROWS = 64
DST_IMG_COLS = 64

MIN_STR_ANGLE = 0.1



def aug_intensity(image):

    #Converting to HSV color space to isolate intensity channel
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    #converting from int to floating point for math operations
    img = np.array(img, dtype=np.float64)

    #generating random float in the range of [0.5, 1.5)
    rand_flt = np.random.uniform() + 0.5

    #scalling the V channel by the random float
    img[:, :, 2] = img[:, :, 2]*rand_flt
    img[:,:, 2][img[:,:,2] > 255] = 255 #addressing overflows

    #converting back to unsigned int before the color space transform
    img_unt = np.array(img, dtype=np.uint8)

    #Converting back to the original RGB color space
    img_op = cv2.cvtColor(img_unt, cv2.COLOR_HSV2RGB)

    return img_op



def aug_shift(image, str_angle, shift_range):

    #generating a random number for shift in x direction in the range [-shift_range/2, +shift_range/2)
    shift_x = shift_range*np.random.uniform() - shift_range/2

    #calculating the corresponding steering angle shift for the shift_x
    str_new = str_angle + shift_x/shift_range*2*0.2


    #generating a random number for shift in y direction in the range [-Y_SHIFT_RANGE/2, +Y_SHIFT_RANGE/2)
    Y_SHIFT_RANGE = 40
    shift_y = Y_SHIFT_RANGE*np.random.uniform() - Y_SHIFT_RANGE/2

    #creating a transformation matrix for the affine transform
    M = np.float32([ [1, 0, shift_x], [0, 1, shift_y]])

    #shifting the image
    rows, cols, chs = image.shape
    img_op = cv2.warpAffine(image, M, (cols, rows))

    #returnig the shifted image together with the corresponding steering angle
    return img_op, str_new


#This function takes as image 160x320 and with 50% probability makes a shadow by
#redcuing the intensity of pixels by half in a randomly chosen region
#That region could be either above or below the line joining points1 and points2.
def aug_shadow(image):

    #Shadow is applied only with a 50% of probability
    if np.random.uniform() < 0.5 :

        return image # no shadow is applied, just return the input image

    else :

        #Point1 = (x1, y1), where x1=0 and y1 a random number in range of [0, 320).
        #Point1 is a point along the top edge of the image
        p1_x = 0
        p1_y = 320*np.random.uniform()

        #Point2 = (x2, y12, where x2=160 and y2 a random number in range of [0, 320).
        #Point2 is a point along the bottom edge of the image
        p2_x = 160
        p2_y = 320*np.random.uniform()


        #generating cordinates for pixels and seperating cordinates of X and Y
        grid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        x = grid[0]
        y = grid[1]

        #creating a mask  (initialized to 0) to decide which pixels gets the shadow
        mask = 0*image[:, :, 0]

        #setting 1 to mask for pixels below the line connecting point1 and point2
        mask[ (x - p1_x)*(p2_y - p1_y) - (p2_x - p1_x)*(y - p1_y) > 0 ] = 1


        #converting to the HLS color space to isolate the light channel
        img_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        scale_down = 0.5
        above_line = mask == 0
        below_line = mask == 1

        #deciding whether to add shadow to the region above the line or below the line
        if np.random.uniform() < 0.5 :
            #adding shadow to the region below the line
            img_hls[:, :, 1][below_line] = img_hls[:, :, 1][below_line]*scale_down

        else :
            #adding shadow to the region above the line
            img_hls[:, :, 1][above_line] = img_hls[:, :, 1][above_line]*scale_down


        #converting back to the RGB space
        img_op = cv2.cvtColor(img_hls, cv2.COLOR_HLS2RGB)

        return img_op



#crops the irrelevant top and bottom rows of the image and resize it to fit the CNN 
def crop_resize_image(image):

    src_rows, src_cols, chs = image.shape

    #cropping 1/5th of the top rows and bottom 25 rows out of the image
    img = image[int(src_rows/5):(src_rows -25), 0:src_cols, :]

    #resizing the image to get a 64x64 one for CNN input
    img_op = cv2.resize(img, (DST_IMG_COLS, DST_IMG_ROWS), interpolation=cv2.INTER_AREA)

    return img_op


#Reads the data file one line at a time and prepares one training 
#instance (image and the y value) at a time
#It randomly selects an image from left, center of right camera and apply 
#augmentation; then randomly flips it horizontally
def read_image_augment(file_line):

    #picks the image either from left, center or right cameras stream with an equal probability
    rand_flt = np.random.uniform()

    if rand_flt < 0.33 :

        #picking left
        img_path = file_line[1].split()[-1]
        str_shift = 0.25


    else :

        if rand_flt >= 0.33 and rand_flt < 0.67 :

            #picking center
            img_path = file_line[0].split()[-1]
            str_shift = 0.0

        else :

            #picking right
            img_path = file_line[2].split()[-1]
            str_shift = -0.25


    # reading the steering angle and shifting it to correspond to the randomly
    #picked camera stream (left, center or right)
    str_angle = float(file_line[3]) + str_shift


    #reading the image from the path
    img_path = "./" + img_path
    image = cv2.imread(img_path)

    #convert to RGB color space (OpenCV default is BGR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    #applying horizontal and vertical shifts
    X_SHIFT_RANGE = 100
    img_shifted, str_shifted = aug_shift(img, str_angle, X_SHIFT_RANGE)

    #applying intensity changes
    img_inten = aug_intensity(img_shifted)

    #applying random shadow
    img_shadow = aug_shadow(img_inten)

    #Cropping and resizing
    img_cropped = crop_resize_image(img_shadow)

    img_op = np.array(img_cropped)

    #flipping horizontally with a probabilty of 50%
    rand_flt2 = np.random.uniform()

    if rand_flt2 < 0.5 :
        img_op = cv2.flip(img_op, 1)
        str_shifted = -1.0*str_shifted

    return img_op, str_shifted



#Generates a batch of data by reading data and augmenting them
def generate_a_batch(data_lines, batch_size=32) :


    #place holder for the batches the generator would produce at each call to it
    img_op_batch = np.zeros((batch_size, DST_IMG_ROWS, DST_IMG_COLS, 3))
    str_op_batch = np.zeros(batch_size)

    num_total_samples = len(data_lines)

    while 1 : # To loop forever so the generator never terminates

        for op_sample_ind in range(batch_size) :

            rand_sample_ind = np.random.randint(num_total_samples)
            data_line = data_lines[rand_sample_ind]

            #initializing the flag which decides to keep the augmented image of
            #the data_line instance or generate a new one, based on the
            #steering angle with some randomness
            keep_prob = 0

            #Iterating the image augmentation process for augmented instances
            #with steering angle < MIN_STR_ANGLE
            while keep_prob == 0 :

                img_aug, str_angle_aug = read_image_augment(data_line)

                #if the steering angle of the augmented sample if less than
                #MIN_STR_ANGLE deciding to keep it or generate a new one based
                #on keep_prob_T randomly
                if abs(str_angle_aug) < MIN_STR_ANGLE :
                    rand_prob = np.random.uniform()

                    # the dicision to keep the lower steering angle sample is made
                    # based on the keep_prob_T and the random number rand_prob
                    if rand_prob > keep_prob_T :
                        keep_prob = 1

                #When the augmented steering angle is greater than the MIN_STER_ANGLE
                #that augmented sample is kept and added to the outpit batch
                else : 

                    keep_prob = 1

            img_op_batch[op_sample_ind] = img_aug
            str_op_batch[op_sample_ind] = str_angle_aug


        yield (img_op_batch, str_op_batch)









#==Reading data, augmenting via generator and feeding to the CNN and training==

#Initializing a list to hold lines (that contain data samples) in the CSV files
samples = []

#reading the CSV file and appending lines to the samples list
with open('driving_log.csv') as csv_file :

    csv_reader = csv.reader(csv_file)


    i = 0
    #appending lines to the list, one line at a time
    for line in csv_reader :

        if i == 0:
            print("by passing the header row in the CSV file")
            i +=1
            continue

        samples.append(line)
        i +=1

    print("parsed {} original data instances from the CSV file".format(i))



#Initializing keep_prob_T
keep_prob_T = 1

batch_size = 256

num_epochs = 8

num_channels = 3


TRAIN_SAMPLES_PER_EPOCH = 200000

VALID_SAMPLES_PER_EPOCH = 40000


#============difining the model in Keras=================================
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU

#initializing a sequential model in Keras
model = Sequential()

#Lambda layer to normalize the pixels to the range of [-1.0, 1.0]
model.add(Lambda(lambda x: x/127.5 - 1.0,
          input_shape=(DST_IMG_ROWS, DST_IMG_COLS, num_channels),
          output_shape=(DST_IMG_ROWS, DST_IMG_COLS, num_channels) ) )




#colorspace selection layer (3 1x1 convilution filters)
model.add(Convolution2D(3,1,1))


#1st convolution block
model.add(Convolution2D(32,3,3))
model.add(ELU())
model.add(Convolution2D(32,3,3))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))


#2nd convolution block
model.add(Convolution2D(64,3,3))
model.add(ELU())
model.add(Convolution2D(64,3,3))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))



#3rd convolution block
model.add(Convolution2D(128,3,3))
model.add(ELU())
model.add(Convolution2D(128,3,3))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#flatten layer
model.add(Flatten())

#1st fully connected layer
model.add(Dense(512))
model.add(ELU())


#2nd fully connected layer
model.add(Dense(64))
model.add(ELU())


#3rd fully connected layer
model.add(Dense(16))
model.add(ELU())


#Output
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam') # using MSE for the loss and ADAM


#-------------------------------------------------------------------------




#=============Training the model========================================

for epoch in range(num_epochs):

    #splitting the total available data in to training and validation
    train_samples, valid_samples = train_test_split(samples, test_size=0.2)

    #making generators one each for the training and validation data
    train_generator = generate_a_batch(train_samples, batch_size)
    valid_generator = generate_a_batch(valid_samples, batch_size)



    #training the model
    model.fit_generator(train_generator, samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH,
            validation_data=valid_generator, nb_val_samples=VALID_SAMPLES_PER_EPOCH,
            nb_epoch=1, verbose=1)

    #reducing the keep_prob_T with the epoch number
    keep_prob_T = 1/(epoch +1)


#saving the trained model
model.save('model.h5')
