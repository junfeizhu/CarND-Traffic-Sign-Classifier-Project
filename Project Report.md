# Traffic Sign Classifier

## Data Set Summary & Exploration
I used the pandas library to calculate summary statistics of the traffic signs data set:
Each traffic sign image is a 32 by 32 image. There are 43 unique classes in the data set. There are 34799 examples in the training set, 4410 examples in the validation set and 12630 examples in the test set. Then I investigated the training set further. The training set is very imbalance. Class 2, Speed limit (50km/h)
 , has the most examples, which is 2010. However, class 37, Go straight or left, has the least examples, which is only 180. The bar chart for the number of examples of each class is shown below:
 
 ![barchart](https://note.youdao.com/favicon.ico)
 
 Then I also plotted 10 random images for each class to check how do these traffic signs actually look like. You can check my python notebook for these images. Below is just a part of them:
 
 ![traffic_sign_images](https://note.youdao.com/favicon.ico)
 
##  Design and Test a Model Architecture
As a first step, I decide to generate additional data because as mentioned earlier, some classes have extremly few examples in the training set. Then I generated 10 more times of images for each class. I used rotation, translantion and shear operations to generate new images. All these operations are wrapped in the function 'transform_image()'. This functions is inspired by [this post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) recommended by my mentor. Here is an example of an original image and an augmented image:
![data_aug](https://note.youdao.com/favicon.ico)

Then I transformed the image to grayscale and normalized the image so that the number in each pixel is between 0.1 and 0.9. This will make the numerical computation more stable.

I mimiced the LeNet to make my own model for this project. My model is very similar to the LeNet except that I added one more convolution layer to the LeNet. I did this because traffic signs are more complicated than handwriting digits. Thus the model may need more convolution layer to capture those features. The description of my model structure is listed below:

|               | Layer     | Description |
|---            |---        |---          |
|               |Input       | 32X32X1 grascale normalized image|
|  Layer 1      | Convolution 3x3 |1x1 stride, valid padding, outputs 30x30x24 |
|               | Relu       |               |
|               |Max pooling |2x2 stride, outputs 15x15x24|
|               |dropout     |                  |
|   Layer 2     | Convolution 2x2 |1x1 stride, valid padding, outputs 14x14x36 |
|               | Relu       |               |
|               |Max pooling |2x2 stride, outputs 7x7x36|
|               |dropout     |                  |
|   Layer 3     | Convolution 2x2 |1x1 stride, valid padding, outputs 6x6x48 |
|               | Relu       |               |
|               |Max pooling |2x2 stride, outputs 3x3x48|
|               |dropout     |                  |
|   Layer 4     | Fully connected |Input = 432  Output = 120 |
|               | Relu       |               |
|               |dropout     |                  |
|   Layer 5     | Fully connected |Input = 120  Output = 84 |
|               | Relu       |               |
|               |dropout     |                  |
|   Layer 5     | Fully connected |Input = 84  Output = 43 |
|               | Relu       |               |
|               |softmax     |                  |


To train the model, mini-batch algorithm is used. I used an Adam optimizer with batch size 256, 100 epochs, learning rate 0.002 and keep probability for dropout 0.8.

After 100 epochs, the training accuracy is 0.962, the validation accuracy is 0.967 and the test accuracy is 0.935

## Test a Model on New Images
Here are five German traffic signs that I found on the web,which are resized to 32 by 32, grayscaled and normalized:

![test_web_images](https://note.youdao.com/favicon.ico)

Here are the results of the prediction:

|Image  |Prediction|
|-----  |----------|
|Speed limit (30km/h)|Roundabout mandatory|
|Stop   |Stop      |
|Road work|Road work|
|Keep right|Keep right|
|Right-of-way at the next intersection|Right-of-way at the next intersection|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. However, I don't know why the Speed limit (30km/h) traffic sign was mis-classified. To me, the image is quite clear and not difficult to classify. Please give me some suggestion.

And the top 5 softmax probabilities for each image are listed below:

|Image 1        ||        Image 2||
|-------|-------|---------|-------|
|Probability|Prediction|Probability|Prediction|
|0.397|Roundabout mandatory|0.999|Stop|
|0.288|Speed limit (50km/h)|0.000|Turn right ahead|
|0.074|No passing for vehicles over 3.5 metric tons|0.000|Roundabout mandatory|
|0.054|No passing|0.000|Speed limit (60km/h)|
|0.040|Speed limit (30km/h)|0.000|Keep left|


|Image 3        ||        Image 4||
|---------|-------|---------|-------|
|Probability|Prediction|Probability|Prediction|
|0.957|Road work|0.999|Keep right|
|0.020|Children crossing|0.000|Roundabout mandatory|
|0.013|Bicycles crossing|0.000|Turn left ahead|
|0.006|Bumpy road|0.000|Priority road|
|0.001|Slippery road|0.000|Go straight or right|

|   Image 5 ||
|-----------|---------|
|Probability|Prediction|
|0.533|Right-of-way at the next intersection|
|0.236|Beware of ice/snow|
|0.184|Children crossing|
|0.039|Slippery road|
|0.006|Double curve|

For Image 1, the right class Speed limit (30km/h) has only a small probability of 0.04.

## Further thinking
During this project, I did realize two interesting problems.
The first problem is about the imbalance of the training set. Instead of generating 10 more times of images for each label, I also tried to generate more images for labels with few images to make the trainng set balance. However, with such training set, the validation accuracy is only between 0.92 and 0.93.

The second problem is about the training accuracy and validation accuracy. If you check my jupyter notebook, you may realize that for many epochs, especially the first several epochs, the validation accuracy is higer than the training accuracy. But intuitively, it should be the opposite.

Do you have any suggestion about these two problems?