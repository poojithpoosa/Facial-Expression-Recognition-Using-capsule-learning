# Facial-Expression-Recognition-Using-capsule-learning

A person's judgments and discussions on numerous topics are heavily influenced by the emotions that have developed on their face. Surprise, fear, disgust, anger, happiness, and sorrow are some of the most common emotional states that may be categorised according to psychological theory. These emotions may be automatically extracted from facial photos to aid in human-computer interaction and a slew of other uses. Deep neural networks, in particular, can identify the collected patterns and learn complicated properties. In this project I have implemented capsule network model using CK+ dataset. The dataset is converted to gray scale and pre-processed using adaptive histogram equalization. The model showed accuracy of 98.6% on training set and 98.04% on testing set.

## Methodology

![image](https://user-images.githubusercontent.com/61981756/199006875-9778a13d-58f8-4126-a83f-3bead619dd66.png)

Dataset: 
The Extended Cohn-Kanade Dataset (CK+)  was launched in 2010 to replace the first edition of the database. 107 sequences of emotional transitions and 26 participants were included in this version. Non-posed photographs were also captured this time, and each image got a nominal label based on the subject's opinion of the seven main emotion categories, which are Anger; Contempt; Disgust; Fear; Happy; Sadness; and Surprise.

Preprocessing 

* Gray scale conversion
* Histogram Equalization

Classification

* Capsule network

The new architecture of neural networks is based on capsule networks (CapsNet), a more sophisticated approach to prior neural network designs, especially for computer vision applications. This is the first time that CNNs have been employed for computer vision. Even while CNNs have improved greatly in terms of accuracy, there are still significant drawbacks.

* Drawback of Pooling Layers
Convolutional neural networks (CNNs) were originally designed to identify pictures by using convolutions and pooling. Using a convolutional block's pooling layer, it is possible to minimise the data dimension while still achieving spatial invariance, which implies that the item can be recognised and classified wherever in the picture. Despite the fact that this is a fantastic idea, there are some downsides. For example, while doing picture segmentation and object recognition, the pooling process tends to lose a significant amount of data. Object recognition and segmentation become much more difficult when the pooling layer lacks the necessary spatial information, such as rotation, location, scale, and other positional properties. Reconstruction is a time-consuming and error-prone procedure in contemporary CNN architecture, despite its ability to use sophisticated algorithms to restore positional information. If the item's location is slightly adjusted, it doesn't appear to affect its activation percentage, which results in high accuracy in picture categorization but bad performance if you want to find precisely where the object is in the image.

![image](https://user-images.githubusercontent.com/61981756/199007427-f622e789-c78a-4a56-b5e7-d2e6eb48652d.png)

The CNN can classify the above image as face, but this can be mitigated by capsule network.
Geoffrey Hinton devised the capsule network as a solution to these problems . This collection or group of neurons stores information about the object it is attempting to identify in a given image; this information primarily relates to the object's position and rotation in a high-dimensional vector space (8 or 16 dimensions), with each dimension representing something unique about the object that can be intuitively understood.
A rendering in computer graphics is the process of taking into consideration an object's location, rotation, and scale in order to produce a picture on the screen. In contrast to this, our brains use inverse graphics, which is a term for the reverse of this method. When we look at anything, we tend to breakdown it into a series of hierarchical sub-components and form a link between these pieces. Because of this, our recognition of things is independent of the perspective or orientation they are in. Capsule networks are based on this idea.
Let's look at the architecture of a capsule network to see how this works. There are three basic sections to the capsule network architecture, and each portion contains sub-operations.

*	Primary capsules
*	Higher layer capsules
*	Loss calculation

![image](https://user-images.githubusercontent.com/61981756/199007685-2fbf7b40-5182-4dff-83fa-d73ee8e5233e.png)

Workflow:

![image](https://user-images.githubusercontent.com/61981756/199007720-e50aadb3-ac04-40bf-a382-114081d2ee2f.png)

### Results

![image](https://user-images.githubusercontent.com/61981756/199007782-bdfb5a7b-1945-4569-8825-98ee190730fa.png)

![image](https://user-images.githubusercontent.com/61981756/199007840-30f912eb-d72b-4917-bd03-b41fd6685468.png)

 
