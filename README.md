# Introduction
LoopNetAE:An Attention-Augmented Deep Convolutional Network with Ensemble Boosting for Chromatin Loop Detection.
# Installation
LoopNetAE is developed and tested on Linux machines and relies on the following packages:
```
scikit-learn  
tensorflow
keras
hic-straw
joblib  
numpy 
scipy   
pandas 
h5py   
cooler 
joblib
tqdm
```
Create an environment.
```
git clone https://github.com/yimuhuashui/LoopNetAE.git 
conda create -n LoopNetAE python=3.6.13   
conda activate LoopNetAE    
```
# Usage
## Model training
The dataset required by the model can be downloaded in the supplementary material, and the operation of dividing the dataset and positive and negative samples is in the train.py, and run:  
```
python train.py
```
If you want to change the training chromosomes and parameters, you can modify them by train.py the parameters in the code file,Example:
```
    path = "/public_data/yanghao/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
    output = "/home/yanghao/Loopnetae/model-chr15/"
    bedpe = "/home/yanghao/Loopnetae/data/gm12878_ctcf_h3k27ac.bedpe"
```
"path" is the path to the hic data file, "output" is the output path of the model, and "bedpe" is the path to the bedpe data file.
## Model predicting
By predict.py prediction of the trained model, candidate chromatin loops are obtained, and run:  
```
python predict.py 
```
Example of modifiable parameters:
```
    output = "/home/yanghao/Loopnetae/candidate-loops/"
    model = "/home/yanghao/Loopnetae/model-chr15/"
    path = "/public_data/yanghao/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
```
"output" is the output path of the candidate chromatin ring, "model" is the save path of the trained model, and "path" is the path of the hic data file.
## Clustering
Run the clustering.py file to perform clustering and obtain the final chromatin loop file.
```
python clustering.py 
```
Example of modifiable parameters:
```
    infile = "/home/yanghao/Loopnetae/candidate-loops/chr15.bed"
    outfile = "/home/yanghao/Loopnetae/loops/chr15.bedpe"
    threshold = 0.93
```
"infile" is the path for the candidate chromatin loop file, "outfile" is the path for the final chromatin loop, and "threshold" is the clustering threshold.
##  Model testing 
Model testing can be performed using the test.py file, and the output results will include evaluation metrics such as model accuracy and recall, and run: 
```
python test.py
```
The variable parameters are test_chrom, model_dir, test_data_path, and report_dir. Here, test_chrom refers to the test chromosome, model_dir denotes the model saving path, test_data_path indicates the path of the saved test dataset, and report_dir is the storage path for the generated report.
