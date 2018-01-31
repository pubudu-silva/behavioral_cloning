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





