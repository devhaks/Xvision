이 프로젝트는 https://github.com/ayush1997/Xvision 을 fork 하여 간단히 수정 내용이 포함되어 있습니다.

# How to test

## 준비사항

### git 프로젝트 다운로드

```bash
$ git clone https://github.com/devhaks/Xvision.git
```

### 파이썬 로컬용 가상화 패키지 설치

```bash
# 파이썬 버전 확인
$ python --version

# python 2.7 인 경우
$ pip install virtualenv 

# python 3.5 인 경우
$ pip3 install virtualenv

# 파이썬 패키지 매니저 버전 확인
$ pip --version

# python 2.7 인 경우
$ sudo pip install virtualenv

# python 3.5 인 경우
$ sudo pip3 install virtualenv
```

이제 파이썬 `virtualenv` 패키지로 로컬에 파이썬 개발 환경을 구성할 수 있다.

```bash
# 다운받은 프로젝트 디렉토리에 파이썬 로컬 실행 환경 셋팅하기
$ virtualenv Xvision

# 디렉토리 확인하기. bin, lib 디렉토리가 생성되면 정상!!
$ cd Xvision
$ ls -al
```
### 가상모드 실행하기

```bash
$ source ./bin/activate

# 커맨드 앞부분이 변경되었으면 정상적으로 로컬 실행이 완료돰. 
(Xvision) ➜  Xvision $
```

### 파이썬 웹 실행 툴인 주피터 설치하기
[주피터 설치](https://jupyter.org/install) 를 통해 python 버전에 맞게 설치를 진행한다. 다운로드 받은 프로젝트에 설치를 해야한다.

아래 명령을 실행하면 주피터 브라우저가 띄워지고 프로젝트 디렉토리 목록이 보일 것이다.
```bash
(Xvision) ➜  Xvision $ jupyter notebook
```

`scraper/process.ipynb` 파일로 이동하여 `projectRootPath` 이라고 표시된 부분을 프로젝트 경로로 대체하여 저장 한다.

이제 실행해야할 차례인데, 실행전에 `DeepLearning/images` 디렉토리가 있어야 하고 training 이미지 파일이 있어야 한다. 다운로드 방법은 아래의 [Steps to follow](#steps-to-follow) 부분을 참고한다.

images 디렉토리에 이미지를 준비해 놓고 주피터 화면으로 이동하여 `Run` 버튼을 눌러 한 단계씩 진행한다.

만약, 실행준에 모듈을 찾을 수 없다는 에러가 출력된다면 간단히 설치를 해주면 된다.
```bash
(Xvision) ➜  Xvision $ pip install <패키지명>
```

마지막까지 정상 실행 된다면 `DeepLearning` 디렉토리에 아래와 같은 결과물이 생성될 것이다. 
1. final_test_images_calc_nodule_only 
2. final_train_images_calc_nodule_only

----

# Xvision

Chest Xray image analysis using **Deep Learning** and  exploiting **Deep Transfer Learning** technique for it with Tensorflow.

The **maxpool-5** layer of a pretrained **VGGNet-16(Deep Convolutional Neural Network)** model has been used as the feature extractor here and then further trained on a **2-layer Deep neural network** with **SGD optimizer** and **Batch Normalization** for classification of **Normal vs Nodular** Chest Xray Images.

## Nodular vs Normal Chest Xray
<img src="https://github.com/ayush1997/Xvision/blob/master/image/node.jpg" width="300" height="300" />
<img src="https://github.com/ayush1997/Xvision/blob/master/image/normal.jpg" width="300" height="300" />

## Some specifications

| Property      |Values         |
| ------------- | ------------- |
| Pretrained Model | VggNet-16  |
| Optimizer used  | stochastic gradient descent(SGD)  |
| Learning rate  | 0.01|  
|Mini Batch Size| 20 |
| Epochs | 20 |
|2 Layers| 512x512 |
|GPU trained on| Nvidia GEFORCE 920M|

## Evaluation
### Confusion Matrix and Training Error Graph

<img src="https://github.com/ayush1997/Xvision/blob/master/image/cfm.jpg" width="450" height="400" />
<img src="https://github.com/ayush1997/Xvision/blob/master/image/nodule.jpg" width="400" height="400" />

|     |  **Normal** | **Nodule** |
|------|---------|---------|
| **Precision**| 0.7755102| 0.55555556 |
|**Recall**| 0.76 | 0.57692308 |

**Accuracy** : **69.3333 %**

## DataSet
[openi.nlm.nih.gov](https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m=1&n=101) has a large base of Xray,MRI, CT scan images publically available.Specifically Chest Xray Images have been scraped, Normal and Nodule labbeled images are futher extrated for this task.

## How to use ?
The above code can be used for **Deep Transfer Learning** on any Image dataset to train using VggNet as the PreTrained network. 
### Steps to follow 

1. Download Data- the script download images and saves corresponding disease label in json format.

  ```python scraper.py <path/to/folder/to/save/images>```

2. Follow the ```scraper/process.ipynb``` notebook for Data processing and generate

  * Training images folder - All images for training
  * Testing images Folder - All images for testing
  * Training image labels file - Pickled file with training labels
  * Testing image labels file - Pickled file with testing labels

3. Extract features(**CNN Codes**) from the **maxpool:5** layer of PreTrained CovNet(VggNet) and save them beforehand for faster training of Neural network.

    ```python train.py <Training images folder> <Testing image folder> <Train images codes folder > <Test images codes folder>```
    
    * Train images codes folder - Path where training images codes will be stored
    * Test images codes folder - Path where testing images codes will be stored
    

4.  The extracted features are now used for training our **2-Layer Neural Network** from scratch.The computed models are saved as tensorflow checkpoint after every **Epoch**.

    ```python train_model.py <Training images folder> <Train images codes folder> <Training image labels file> <Folder to         save models>```

5.  Finally the saved models are used for making predictions.Confusion Matrix is used as the Performance Metrics for this classifcation task.

    ```python test_model.py <Testing images folder> <Test images codes folder> <Testing image labels file> <Folder with saved models>```
    
    
    
## Some Predictions

![Alt text](https://github.com/ayush1997/Xvision/blob/master/image/pred.jpg "Optional Title")

## References

> 1. [Learning to Read Chest X-Rays: Recurrent Neural Cascade Model for Automated Image Annotation](https://arxiv.org/pdf/1603.08486.pdf)

> 2. [Deep Convolutional Neural Networks for Computer-Aided Detection: CNN Architectures,
Dataset Characteristics and Transfer Learning](https://arxiv.org/pdf/1602.03409.pdf)

## Contribute

If you want to contribute and add new feature feel free to send Pull request [here](https://github.com/ayush1997/Xvision/pulls) :D

To report any bugs or request new features, head over to the [Issues page](https://github.com/ayush1997/Xvision/issues)

## To-do

- [ ] Implement saliency map or use Deconv for better visualizations. 
