**Steering for Self-Driving Cars**

This project implements a convolutional neural network (CNN) that takes a front facing camera images as input and outputs a steering command (angles). The CNN was tested and trained in a simulator environment. The simulator [1] was created by Udacity using the Unity engine that uses real game physics to create an approximation to real driving conditions. The CNN implemented is a model described by Nvidia in a research paper published April 2016 [2].

**CNN Architecture**

The model described by nvidia [2] uses a sequence of 5 convolutional layers followed by a sequence 4 fully connected (dense) layers with ELU layers in between for activation. ELU as opposed to regular RELU activation fires -1 for values less than 0 helping the network learn faster [3]. Additionally we introduce dropout layers in the fully connected portion of the network to run experiements to see if dropouts parameter may be an option in addressing overfitting concerns. The input to the network is an image of shape (66, 200, 3) and the output of the network is a normalized steering angle value between 0.0 and 1.0 that the Udacity simular understands how to interpret (de-normalize).

Convolution2D(?, 31, 98, 24) --> ELU --> Convolution2D(?, 14, 47, 36) --> ELU --> (?, 5, 22, 48) --> ELU --> Convolution2D(?, 3, 20, 64) --> ELU --> Convolution2D(?, 1, 18, 64) --> ELU --> Dense(?, 1164) --> ELU --> Dropout --> Dense(?, 100) --> ELU --> Dropout --> Dense(?, 50) --> ELU --> Dropout --> Dense(?, 10) --> ELU --> Dropout --> Dense(?, 1)

Total number of trainable params: 1595511

**Data, Data Collection and Normalization**

The car driving simulator can be run in 2 modes - Training Mode and Autonomous mode, and presents 2 tracks where the car may be driven. 'Training mode' as the name implies is used to gather data to be fed in to the network for training purposes. The output of training mode is a csv file and a series of images from the front left, center and right cameras and the current steering angle at that particular point in time. The image captured can vary in quality depending on the "Graphics Quality" setting presented in the simulator. The options are fastest, 'fast', 'simple', 'good', 'beautiful' and 'fantastic' with 'fastest' requiring the least amount of compute power to fantastic requiring the most. 'Fastest' for example will generate images/scenes with the least amount of detail such as shadows and 3D effects while 'fantastic' will include the most amount of details. The network architecture ideally should be able to handle images of all qualities and produce good steering commands (we'll discuss how well the model did in this exercise below).

All images fed in to the network were first normalized as follows for the network learn and to reach its global optimum quickly.

Source Image -> Isolate a region of interest -> Min/Max Normalization -> Mean Subtraction -> Image Resize (height x width)

Region of interest blacked out a small portion at the top of the image (the horizon) past the visible end of the road and bottom portions of the screen to remove the visible engine compartment of the car without removing any the information about the current road surface and the road ahead for the network to learn steering commands. Finally after the min/max and mean substration normalization steps the image was resized to the input size used by the nvidia model (66x200).

```
        image_out = Data.__region_of_interest(image)
        image_out = Data.__pixel_normalize(image_out)
        image_out = Data.__pixel_mean_subtraction(image_out)
        image_out = Data.__resize_image(image_out, Data.INPUT_IMAGE_HEIGHT, Data.INPUT_IMAGE_WIDTH)
```

Data collection can be extermely tedious using the available simulator and the data quality low especially without a proper gaming joystick controller. Keyboard inputs can be erratic and result in data that teaches the network to drive erratically and off the road and off center rather than drive steadily along the center of the lane and for this reason training was done purely on the high quality data set provided. The downside to using the provided dataset is that you end up relying heavily on augmented data to teach the network to drive under various circumstances including exceptional circumstances when the car must recover and correct itself from a previous steering mistake (stretch goal at a later date for this exercise would be to use an agile trainer [4] coupled with a joystick controller to train our network much more precisely). We will discuss regularization/augmentation below.

**Regularization to prevent overfitting**

The data set available for training was augmented to produce additional images and steering commands as follows:

> 1. Select a camera angle at random (and adjust the steering angle of the center camera for left and right camera views). This was done at random with a 1/3 chance of each camera being picked for every image selected for a training batch. For the left and right cameras we add a +0.3 and -0.3 angles to the center camera steering angle value.
> 2. Flip the images left to right (and invert the steering angles when we do so). This was done at random with a 50:50 chance of being flipped for every image selected for a training batch.
> 3. Adjust brightness (+/-) of the image by a small value with 50:50 odds of the brightness value being adjusted vs. the original source brightness being picked. Adjusting brightness will teach the model to ignore shadows and various lighting conditions.
> 4. Shift the image left-right and/or top/bottom by a number of pixels to simulate the effect of the car being off center and driving up or down hill. Adjust the steering angle as a % of the pixels shifted horizontally. The steering angle value used here was empirically determined by testing on track #1.
>5. Furthermore, more we split the data in to a training and validation set and we made this a configurable parameter under settings.ini. We also made the number of samples per epoch configurable given that our augmentation process randomly selects and modifies images. This has the effect of producing a larger regularized dataset per epoch minimizing overfitting conerns.

settings.ini
```
samples_per_epoch_multiplier: 8
training_split: 0.8
validation_split: 0.2
```

**Training**

Training was done on a AWS EC2 p2.xlarge (1 x Nvidia Tesla K80) instance. A Keras fit generator is used with a worker count equal to the number of processors/cpu cores available (on a p2.xlarge this value was 4) on the training machine to improve memory usage and performance so that not all the data used train needed to be pre-generated and held permanently in memory for the duration of the training execercise. Each epoch may use an independent worker thread to generate its training batches on the CPU.

```
        workers = multiprocessing.cpu_count()
        logging.info("Set fit generator. workers = %s optimizer: %s" % (workers, self.__optimizer))

        self.__model.compile(loss="mse", optimizer=self.__optimizer) # , metrics=['accuracy'])
        self.__model.fit_generator(training_data.fit_generator(), validation_data=(valid_images, valid_angles), samples_per_epoch=training_data.samples_per_epoch(), nb_worker=workers, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpoint], pickle_safe=True)
```

The model uses checkpoints to regularly save the weights that result in the lowest validation loss and once finally after all the epochs are complete. On subsequent training iterations (with or without various adjustments made to the hyper parameters) we use transfer learning and use previously saved weights as the starting point so that the entire training exercise from the very begining with randomly initialized weights need not be repeated on each training run.

Example of the command line client with arguments to start or continue training:

```sh
python steering.py -cmd train -data ./data/set1/driving_log.csv -epoch 5 -loadsaved True
```

**Autonomous mode results: Track 1 & 2**

The model was trained for about 15 epochs in total, 5 epochs at a time at most. After each training exercise the resulting weights was tested in the simulator and the decision to continue training or augment various hyper params was based on the observed results. Transfer learning aspect of the constructed model allows us to re-use a previously saved model weights to short circuit our total learning time. Training speeds on p2.xlarge instances were remarkably fast (~60 seconds to process 50,000 images) compared to a current generation high end macbook pro.

Using dropout values > 0.0 did not appear to significantly improve results on Track 1. So dropout value is left at 0.0. Its is also notable that the nvidia research paper does not mention the use of dropout layers and just simple regularization techniques alone appear sufficient for this network architecture and data sets.

The resulting trained network is able to successfully navigate track 1 repeatedly (in Good quality graphics setting) and the car navigates Track 2 successfully even though the model wasn't trained on this track at all but only at lower quality (fastest) graphics settings that avoids shadows. Problems noticeable during driving (as visible in the videos linked below):

> 1. Areas where there are patches of shadows casted on the road cause the network to make minor steering adjustments. While these adjustments are only minor they are not ideal. Shadows are particularly a problem on track 2 using "Good" quality graphics settings (Car navigates Track 2 successfully on lower quality graphics settings with no shadows rendered).
> 2. There is one area right before the bridge section in Track 1 where the car drives right on top of the lane boundary line on the left. The network corrects this pretty quickly and recovers but this is less than ideal and very unnerving to a human passenger on board.
> 3. Speed/Throttle were hard coded in autonomous mode. This is again is not ideal and ideally the network should take in to consideration (take as input) the current speed of the vehicle. Speed relative to sampling rate has an impact on how much (steep) the steering angle should be in situations where corrections are necessary. This speed vs. steering angle issue is particularly noticeable navigating track 2 where there are lots of up/downhill sections.

See the section on *further Improvements* below on some possible ways to fix these two issues we've noticed.

Running the predictions server:
```
python drive.py ./saved/model.json
```

Track 1 autonomous mode:

[![IMAGE ALT TEXT](http://img.youtube.com/vi/0-5QSHPL6yg/0.jpg)](http://www.youtube.com/watch?v=0-5QSHPL6yg "Track 1 Autonomous mode Results")

Track 2 autonomous mode:

[![IMAGE ALT TEXT](http://img.youtube.com/vi/1gzcvoJ_iOg/0.jpg)](http://www.youtube.com/watch?v=1gzcvoJ_iOg "Track 1 Autonomous mode Results")


**Further improvements**

> 1. Augment the data further by randomly adding shadows on the road to train the model to ignore these features.
> 2. Use an agile training simulator to take control of the car as it approaches the bridge section to teach it to drive straigh along this section.
> 3. Consider current speed in the network to fine tune steering commands. The obvious downside here is that now we have increase the number of input and trainable parameters. So we'll need a larger training data set. Alternatively, we could try increasing the sampling (prediction) rate to account for speed (so make lots and lots of small adjustment descisions rather than fewer but larger steering angle corrections).

These 3 improvements require some additional work to augment the image processing pipeline and implement the agile training simulator described in [4].

**Project Folder Structure**

```
├── steering.py             # Command line tool for training & predictions
├── steering
│   ├──  data.py            # Training data (encapsulates image preprocessing  pipeline)
│   ├──  model.py          # Neural network model
├── settings.ini            # model settings file
├── drive.py                # drive server for predicting steering angles
├── data                    # training data sets
│   ├──  set1
├── saved                  # saved model and weights
│   ├──  model.json        # saved model
│   ├──  model.h5          # saved weights
```

**Steering .py command line usage**

```
usage: steering.py [-h] [-data DATA] -cmd {selfdiag,train,predict}
                   [-epoch EPOCH] [-loadsaved] [-v] [-file FILE]
                   [-expected_angle EXPECTED_ANGLE]

Self-Driving Car Steering Model

optional arguments:
  -h, --help            show this help message and exit
  -data DATA            Train data csv file
  -cmd {selfdiag,train,predict}
                        Commands (default: selfdiag)
  -epoch EPOCH          Training Epochs (Default 5)
  -loadsaved            Load saved weights (default: True)
  -v, --verbose         Verbose output
  -file FILE            Image file to use in predicting (required when cmd ==
                        predict)
  -expected_angle EXPECTED_ANGLE
                        expected steering angle
```

**Settings.ini**

```
[model]
type: nvidia

[training]
batch_size: 256
optimizer: adam
samples_per_epoch_multiplier: 8
training_split: 0.8
validation_split: 0.2
select_camera_at_random: True
generate_new_images: True
dropout_prob: 0.0
```

**References**

[1] Udacity Car Driving Simulator
> * [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587527cb_udacity-sdc-udacity-self-driving-car-simulator-dominique-development-linux-desktop-64-bit-5/udacity-sdc-udacity-self-driving-car-simulator-dominique-development-linux-desktop-64-bit-5.zip)
> * [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587525b2_udacity-sdc-udacity-self-driving-car-simulator-dominique-default-mac-desktop-universal-5/udacity-sdc-udacity-self-driving-car-simulator-dominique-default-mac-desktop-universal-5.zip)
> * [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/58752736_udacity-sdc-udacity-self-driving-car-simulator-dominique-default-windows-desktop-64-bit-4/udacity-sdc-udacity-self-driving-car-simulator-dominique-default-windows-desktop-64-bit-4.zip)

[2] NVIDIA - [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

[3] [ReLU Was Yesterday, Tomorrow Comes ELU](http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/)

[4] [Agile training simulator](https://github.com/diyjac/AgileTrainer)

[5] All the awesome udacity student peers!

