### Implementation Code for exemplary-XBL

This repository contains an implementation source code for the exemplary XBL algorithm. The published paper can be found at [this link](https://link.springer.com/chapter/10.1007/978-3-031-45275-8_26).

To install all requirements: 
```
pip install -r requirements.txt 
```

The employed dataset is published at [Chowdhury et al. 2020](https://ieeexplore.ieee.org/abstract/document/9144185) and [Rahman et al. 2021](https://www.sciencedirect.com/science/article/pii/S001048252100113X?casa_token=vIPeN_Uto3YAAAAA:udtMoz0sXFkvKWoQP8AddkbgOj6FXWxWjxHubEWUKJttXUkCEsqyG3YwBjVleyV8zCrTu90), and it can be downloaded following [this link](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).


### Instructions

Download this repository and change the working directory to exemplaryXBL, and follow the instructions below:
```
cd exemplaryXBL
```

Split the downloaded dataset inside '/dataset/'.

To start training a model using eXBL:
```
python train.py
```
