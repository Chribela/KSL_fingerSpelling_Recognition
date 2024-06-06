# KSL-FINGERSPELLING-RECOGNITION
![ksl image](https://github.com/JamesMbeti/KSL-FINGERSPELLING-RECOGNITION/blob/main/fingerspell.jpeg)


## Table of contents 
- [Business Understanding](#business-understanding)
- [Data preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluations)

---

## 1. Business Understanding
### Overview
Fingerspelling, a technique that utilizes hand formations to represent words and letters, plays a vital role in communication for individuals who are deaf or hard of hearing. However, merging this manual communication method with modern technology, such as smartphones, poses challenges as fingerspelling is typically faster than device recognition. Bridging this gap between fingerspelling and smartphone typing is crucial to improve communication accessibility.

### Problem Statement

The deaf and hearing impaired community faces significant communication barriers with the rest of the society. This is because sign language is not widely understood by everyone else around them, and this can lead to difficulties in communication or communication breakdowns. To address this issue, this project aims to develop a Convolutional Neural Network (CNN) model specifically designed for fingerspelling recognition, allowing for accurate identification of individual letters and complete words in different sign languages. By improving the recognition of fingerspelling gestures, the model seeks to enhance communication accessibility for individuals who are deaf or hard of hearing, promoting inclusivity and fostering effective communication with the broader society.

### Objectives

#### Main objective:

* The main objective is to create an innovative machine learning model that acts as a vital bridge, connecting the deaf and mute community with the wider society by translating fingerspelling images to text.

#### Specific objectives:

* Develop a Convolutional Neural Network (CNN) model specifically designed for fingerspelling recognition in different sign languages.
* Train the model using a large dataset of fingerspelling gestures in various sign languages, ensuring accuracy and reliability in recognizing individual letters and complete words.
* Conduct extensive testing and evaluation to assess the model's performance and accuracy in recognizing fingerspelling gestures across different sign languages.
* Deploy the model.

## 2. Data Understanding

The dataset used in this project consists of fingerspelling gesture images in Kenyan Sign Language (KSL). The dataset is a combination of a publicly available Kaggle dataset and a custom dataset collected specifically for this project.

> kaggle dataset

* The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) 
* The signs provided are based on the Kenyan Sign Language letter database which is made up of 24 classes of letters with the exclusion of J and Z. The two letters have been excluded because they require the use of motion.
* This dataset provided a foundational resource for the project's fingerspelling recognition model.
  

 

> Raw dataset

* To augment the Kaggle dataset and address potential limitations, the project team conducted a separate data collection effort. 
* The custom dataset was collected by the project team themselves, involving the capture of high-quality images of fingerspelling gestures in KSL. 
* This custom dataset aimed to provide a more comprehensive and diverse set of fingerspelling gesture images for training and evaluation.


------
## 3. Data Preparation
Within our data preparation phase, we performed the following tasks:
* Clean Data
* Checking Duplicates
* image processing
* Feature Engineering 


------
## 4. EDA
The following analysis was performed on the data:
* Previewing the images in the data
* univariate Analysis on the labels


------
## 5. Modeling 
The models include;
* Dense neural network
* Convolution neural network
* Google Teachable Machine

-------
## 5. Evaluation 
Our success metrics is accuracy. The model that had the highest accuracy for the validation set was chosen to be the best model and thus used to predict the fingerspelling images.


----
## 6. Challenges

* The dataset was relatively small, which could have limited the accuracy of the model.
* The images in the dataset were not of uniform quality, which could have also affected the accuracy of the model.
* The hand gestures in the dataset were limited to the 24 letters of the Kenyan Sign Language, which means that the model would not be able to recognize fingerspelling for other letters or words.
  
-----
## 7. Conclusions

* Despite the challenges, the model was able to achieve a high accuracy of above 90% on the test dataset.
* The model could be used to improve communication for deaf and hard of hearing individuals, as it would allow them to communicate more easily with people who do not know sign language.

---

## 8. Recommendations

* The dataset could be expanded to include more images of fingerspelling gestures, which would improve the accuracy of the model.
* The images in the dataset could be improved in terms of quality, which would also improve the accuracy of the model.
* The model could be extended to recognize fingerspelling for other letters and words, which would make it more versatile.
* The model could be integrated with other technologies, such as NLP and speech recognition, to provide a more comprehensive communication solution for deaf and hard of hearing individuals.


