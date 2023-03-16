
<div align="center">

# Human_Segmentation_Using_Albumentations💫
---

1.  Understand the Segmentation Dataset and you will write a custom dataset class for Image-mask dataset. Additionally, you will apply segmentation augmentation to augment images as well as its masks. For image-mask augmentation you will use albumentation library. You will plot the image-Mask pair.

2. Load a pretrained state of the art convolutional neural network for segmentation problem(for e.g, Unet) using segmentation model pytorch library.

3. Create train function and evaluator function which will helpful to write training loop. Moreover, you will use training loop to train the model.
   

# 💻Tech Stack <img src = "https://media2.giphy.com/media/QssGEmpkyEOhBCb7e1/giphy.gif?cid=ecf05e47a0n3gi1bfqntqmob8g9aid1oyj2wr3ds3mg700bl&rid=giphy.gif" width = 32px> 
![PYTHON](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen) ![PANDAS](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) ![NUMPY](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=PyTorch&logoColor=white) ![PYTORCH](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![GIT](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white) ![GIT](https://img.shields.io/badge/AWS_S3-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white) ![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white) ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E) ![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)


---

## <img src="https://media.giphy.com/media/iY8CRBdQXODJSCERIr/giphy.gif" width="25"> <b> API</b>


Landing Page of Application
![Screenshot](static/snips/snip1.png)

Interface to upload audio file
![Screenshot](static/snips/snip2.png)




</div>
## How to run?

### Step 1: Clone the repository
```bash
git clone "https://github.com/satyazmx/Human_Segmentation_Using_Albumentations.git" repository
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p env python=3.10 -y
```

```bash
conda activate env/
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Export the  environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

```
Before running server application make sure your `s3` bucket is available and empty

### Step 5 - Run the application server
```bash
python app.py
```

### Step 6. Train application
```bash
http://localhost:8080/train
```

### Step 7. Prediction application
```bash
http://localhost:8080
```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build -t test .

```

3. Run the Docker image
****
```

docker run -d -p 8080:8080 <IMAGEID>
```

🌐  <h3>Infrastructure Required.</h3>
1. AWS S3
2. ECR (Amazon ECR repository)
3. EC2 (Amazon EC2 Instance)
4. GitHub Actions

**Artifact** : Stores all artifacts created from running the application

**Components** : Contains all components of Machine Learning Project
- DataIngestion
- DataTransformation
- ModelTrainer
- ModelEvaluation
- ModelPusher

**Custom Logger and Exceptions** are used in the project for better debugging purposes.


## Take Aways

You now have a better understanding of 
- Albumentations Library that we use for Data Augmentation.
-  UNET Architecture for Image Segentation. 
-  Deep Learning Framework Pytorch.
-  Preparing Custom Data in Pytorch.
-  Webframework like Flask.

=====================================================================