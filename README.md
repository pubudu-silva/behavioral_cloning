# Demonstrating the Power of Behavioral Cloning via Simple Practical Example

Behavioral cloning(BC) is a very powerful concept in machine learning. It is typically used in when one wants to train a machine learning model
to clone a behaviour of some system or person. In order to demonstrate how to BC I am going to use the [Udacity driving simulator](https://github.com/udacity/self-driving-car-sim)
which Udacity is using for one of their assignments in [Self driving car nano-degree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).
The goal of this project is to capture the driving behaviour of human users by recording them driving the car in the simulator and training
a model with the captured data to clone the behavior of the humans, such that the model would aquire driving skills by observing
humans driving. This is also a classic example of end-to-end learning, where machine learning is applied over the full scope of the problem,
without any mediation from the practitioner in going from the learning objective to learning.

The first step is to drive the car manually in the simulator and capture the driving behavior, which will be the training and validation dataset
for this project. Since the learned model's driving is going to be only as good as the person it is trying to immitate, it is important that
that the manual driver does a good job in driving in the simulator. I personally is not a fan of video car games and known to go crazy in
driving a car using only the keyboard. Luckily Udacity has captured a smooth manual driving record and published it for their students doing the
assignment. I am going to use that dataset for this project, to make sure our machine learning model doesn't follow my foot steps in to becoming a bad virtual driver.

## Exploring the dataset
The udacities dataset records four aspects of the driving behaviour (steering angle, throttle, brake and speed) with snapshots taken by three dash cameras (left, center and right). The first three parameters (steering, throttle and break) captures the human drivers driving characteristic while the camera images record what the driver saw when he responded with those three controls. Speed records the speed the vehicle was travellig at that point, which is not a user input/behavior but a direct result of users behavior. The dataset is presented as a CSV file with 7 columns; the first three columns contains the file names of photos taken by center, left and right dash cameras; the last four contains the steering angle, throttle, brake and speed. All the images are included in the 'IMG' directory. You can download this dataset from [this link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

In order to keep it simple, I am going to limit the task of the machine learning model trained in this project to predicting the steering angle given the center camera image as the input. In that sense this BC is just cloning the steering behaviour of the human driver using the only input available for the driver, i.e. the center camera view. However as you will see in our final solution we are going to use the images from left and right cameras for training and validation, so we can tripple the number samples, by treating them as center camera images via adjusting the corresponding steering angle. One can easily use the same concepts to clone other two behaviors (throttle and break) too.

Following are some basic statistics of the dataset that is key in making initial decisions of this project:
* Number of samples : 8036
* The steering angle varies from -1 to +1
* There are lot of zeros observed in the steering angle just by eyeballing.
* In fact the mean of absolute steering angles:0.07, with a varience: 0.01
* The mean of steering angles: 0.004, and skewness: -0.13. Which means there is a left bias in the steering angle. Which makes sense by just observing the track udacity used in the simulator to record this dataset, as it has more left bends than right ones.

Above basic statistics of steering angle reveals that the majority of the time the human driver was not steering at all, and it times she did it was mostly left turns. As you will see later, this will be a key finding when debugging our machine learning module. Following are three camera images for a raondom sample.

Left camera image:
![alt text](/left.jpg "Image from the left dash camera")


Center camera image:
![alt text](/center.jpg "Image from the center dash camera")


Right camera image:
![alt text](/right.jpg "Image from the right dash camera")


## Tring out the most basic solution
As with other projects, I am going to try the most basic solution with minimal work first to see what how far we can go with a minimalistic approach. This provides us an oppertunity to fail fast, prevent over desgining and learn by mistakes to reach the successful solution faster without overdoing it.

Let's scope the problem. We have to train a machine learning model with only input beeing the center camera image, and the only output beeing the steering angle (how much the self driving car should steer in response to seeing that view from the center camera). Since it is an image input, CNN is the natural choice. However instead of classification we have a regression problem here. 
Hence the last fully connected layer will have just one output. In most relatively simpler tasks like this, I start with LeNet kind of a simple CNN. So I went with a CNN with two convolution layers (each with six 5x5 filters, followed by RELU and max pooling) and  followed by three fully connected layers (having 120, 84 and 1 neurons respectively). The input to the network was normalized in the most basic way, i.e. deving by 255 and substracting 0.5. I trained and validated the network with 8036 samples (fitting center camera images to corresponding values of the steering angle) with 20% validation split. I was able to observe indications of overfitting in the training process just by monitoring training and validation loses. I tried the trained model on the simulator, the car drove well in straight protions of the track but couldn't handle larger bends and went out of the rode quickly. 

Let's refine by addressing the overfitting which was observed at the training phase. There could be couple of reasons overfitting.
* ~6.5K training samples (~8K with 80%:20% training validation split) is not sufficient for even this small CNN
* The input image is 320x160 pixels. Large part of the input image, including bottom 1/5th which contains the hood of the car and top 1/3 with include far away horizons, is not relevant to deciding the steering angle. A human driver will only focus on first 10-20 feet away from the front edge of the car to decide on how much to steer. Since we are trying to clone a human drive, its important to limit the focus of our model to roughly the extent that a human driver will limit her focus to. 
* As we observed in statistics there is a left turn bias in the dataset, hence the model is almost gurantee to fail in a right bend especially with this much little data.


The 2nd point can be addressed by cropping the bottom 1/5th and top 1/3rd out of the image before it is fed in to the CNN. The 3rd can be addressed via a simple augmentation trick: duplicating the entire dataset by flipping images horizontally and flipping the steering angle sign. This doubles the dataset and somewhat address the 1st concern too. Re-trained and validate the same network after above steps. This time there was no signs of overfitting from training and validation loss. Tried the model on the simulator. It drove OK for the most part, except for couple spots on the track, where it couldn't recover and went off the road.

Still continuing in diagnosis and reactive mode and keeping to the true spirit of BC, I wanted to teach my model by example. i.e. by recording driving bahavior that would prevent the car from going off the tracks in situations where it failed. I literally took the car to spots where model started to go off tracks and recorded several examples of how I would recover from those situations. Added these new data points to the dataset, followed above augmentation and pre-processing steps with the whole dataset and trained a new model. Tried that model in the track and that did pretty well for couple of rounds in the track as you may see in [this video](www.youtube.com).

While this does the job, there are several issues with this result and the approach.
* The approach I took to correct for situations where it went of the track (by recording more data points to show how to recover in those specific cases) is a solid example of how to overfit a model to specific environment - in this case to the specific track. Those kind of overfitting is not easily observable via traning and validation losses in an end-to-end training situation like this. These are the situations we should use our non-partial judgement about our own approaches.
* Even though the car managed continue drive forward, a closer look at the video reveals that it drove over the curb number of times. Moreover its drive was not smooth as in several of the bends it was struggling to keep in the track.


Hence it is time to take all the above facts in to consideration and start fresh with a bit more sophisticated approach. 
