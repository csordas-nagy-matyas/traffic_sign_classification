**Traffic sign classification with deep learning**

Nowadays, Artificial Intelligence applications can be found almost everywhere, when we are using mobile phones, computers, cars or social media platforms. This time, I would like to focus on the automotive applications with deep learning approaches. I have found a so-called traffic signs dataset on Kaggle, which consists of various types of traffic signs along with their labels. The main objective of this task to build up a suitable deep learning architecture to classify each of the traffic signs.

The dataset what I have chosen can be found on Kaggle: <https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed?resource=download&select=datasets_preparing.py>

**Summary of the dataset:**

Number of samples in the pickles:

- train: 34799
- validation: 4410
- test: 12630

The labels are stored in sparse categorical format in the pickle files, so we have to apply a number to label mapping if we would like to see the meanings of the categories. The mapping table can be found in the label\_names.csv file.

Some of the attributes are listed below:

|**ClassID**|**SignName**|
| :-: | :-: |
|0|Speed limit (20km/h)|
|1|Speed limit (30km/h)|
|9|No passing|
|10|No passing for vehicles over 3.5 metric tons|
|11|Right-of-way at the next intersection|
|12|Priority road|
|13|Yield|
|14|Stop|


**Summary of training three variations of the Deep Learning models I selected:**

|**Model**|**Input Size**|**Batch size**|**Steps per epoch**|**Total/Trainable Parameters**|**Train Acc.**|**Validation Acc.**|**Test Acc.**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|LeNet5\_Custom|60x60|32|500|2,322,431|98\.49%|97\.07%|95\.62%|
|AlexNet\_Custom|130x130|32|300|36 577 739/36 577 115|97\.93%|97\.30%|94\.99%|
|VGG16|120x120|32|500|14 714 688/12 979 200|93\.36%|88\.48%|86\.86%|
|**InceptionV3**|**130x130**|**32**|**300**|**21 802 784/21 768 352**|**98.57%**|**96.67%**|**96.35%**|

**Comparison of the investigated DL models:**

**Custom LeNet5:**

|**Layer**|**Feature Map**|**Size**|**Kernel Size**|**Stride**|**Activation**||
| :-: | :-: | :-: | :-: | :-: | :-: | :- |
|Input|Image|1|60x60|-|-|-|
|1|Convolution|6|60x60|5x5|1|ReLu|
|2|Max Pooling|6|30x30|2x2|2|-|
|3|Convolution|16|30x30|5x5|1|ReLu|
|4|Max Pooling|16|15x15|2x2|2|-|
|4|Convolution|120|15x15|5x5|1|ReLu|
|5|FC|-|84|-|-|ReLu|
|6|FC Dropout(0.1)|-|84|-|-|-|
|7|FC|-|43|-|-|SoftMax|

**Custom AlexNet:**

|**Layer**|**Feature Map**|**Size**|**Kernel Size**|**Stride**|**Activation**||
| :-: | :-: | :-: | :-: | :-: | :-: | :- |
|Input|Image|1|130x130|-|-|-|
|1|Convolution|24|63x63|5x5|2|ReLu|
|2|Max Pooling|24|31x31|3x3|2|-|
|3|BatchNorm.||31x31|||-|
|4|Convolution|48|31x31|5x5|1|ReLu|
|5|Max Pooling|48|29x29|3x3|1|-|
|6|BatchNorm.||29x29|||-|
|7|Convolution|96|29x29|3x3|1|ReLu|
|8|BatchNorm.||29x29|||-|
|9|Convolution|96|29x29|3x3|1|ReLu|
|10|BatchNorm.||29x29|||-|
|11|Convolution|48|29x29|3x3|1|ReLu|
|12|BatchNorm.||29x29|||-|
|13|Max Pooling|48|27x27|3x3|1|-|
|14|FC|-|1024|-|-|ReLu|
|15|FC Dropout(0.1)|-|1024|-|-|-|
|16|FC|-|512|-|-|ReLu|
|17|FC|-|43|-|-|SoftMax|

**VGG 16 with fully connected layers:**

|**Layer**|**Feature Map**|**Size**|**Kernel Size**|**Stride**|**Activation**||
| :-: | :-: | :-: | :-: | :-: | :-: | :- |
|Input|Image||120x120|-|-|-|
|Output|VGG16 output|512|3x3|-|-|-|
|1|FC |-|128|-|-|Relu|
|2|FC|-|43|-|-|SoftMax|

**InceptionV3 with transfer learning:**

|**Layer**|**Feature Map**|**Size**|**Kernel Size**|**Stride**|**Activation**||
| :-: | :-: | :-: | :-: | :-: | :-: | :- |
|Input|Image||130x130|-|-|-|
|Output|InceptionV3 output|2048|2x2|-|-|-|
|1|FC |-|128|-|-|Relu|
|2|FC|-|43|-|-|SoftMax|

**Summary Key Findings and Insights:**

If we focus on the data exploration step, we can determine, that the dataset is strongly imbalanced and the resolution of the pictures are very low. Some of the pictures have too low or too high contrast values. I made some measures to improve the resolution and correct the contrast values. I implemented a balanced\_data\_generator, which can generate balanced batches during training. As a preprocessing step, I applied histogram equalization on the pictures to avoid inappropriate contrast values.

In terms of the deep learning models, the InceptionV3 model had the best accuracy, but it took so much time to train due to roughly 21 million trainable parameters. I was surprised, when I realised if I re-train more layers, the model achieves better score. Finally, I re-trained the entire InceptionV3 model on the traffic signs dataset and that gave me the best performance.
