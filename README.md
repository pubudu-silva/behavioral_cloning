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
assignment. I am going to use that dataset for this project, to make sure our machine learning model doesn't follow my foot steps in to becoming a bad
virtual driver.

## Exploring the dataset
The udacities dataset records four aspects of the driving behaviour (steering angle, throttle, brake and speed) with snap shots taken by three dash cameras (left, center and right). The first three parameters (steering, throttle and break) captures the human drivers driving characteristic while the camera images record what the driver saw when he responded with those three controls. Speed records the speed the vehicle was travellig at that point, which is not a user input/behavior but a direct result of users behavior. The dataset is presented as a CSV file with 7 columns; the first three columns contains the file name of snaps taken by center, left and right dash cameras; the last four contains the steering angle, throttle, brake and speed. All the images are included in the 'IMG' directory. You can download this dataset from [this link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

In order to keep it simple, I am going to limit the task of the machine learning model trained in this project to predicting the steering angle given the three camera image inputs. In that sense this BC is just cloning the steering behaviour of the human driver. One can easily use the same concepts to clone other two behaviors (throttle and break) too.

Following are some basic statistics of the dataset that is key in making initial decisions of this project:
* Number of samples : 8036
* The steering angle varies from -1 to +1
* There are lot of zeros observed in the steering angle just by eyeballing..
* In fact the mean of absolute steering angles:0.07, varience: 0.01

Above basic statistics of steering angle reveals that the majority of the time the human driver was not steering at all. As you will see later, this will be a key finding when debugging our machine learning module. Following is are three camera images for a raondom sample:
![left](/left.jpg)
![center](/center.jpg)
![right](/right.jpg)


