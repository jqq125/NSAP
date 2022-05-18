# NSAP:A neighborhood subgraph aggregation method for drug-disease association prediction
This is our Pytorch implementation for the paper:
Qiqi Jiao,Yu Jiang, Yang Zhang, Yadong Wang, and Junyi Li(2022).NSAP:A neighborhood subgraph aggregation method for drug-disease association prediction.


## Table of Contents

- [Environment Requirement](#requirement)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)



## <span id='requirement'> Environment requirement </span>  

The code has been tested running uder python 3.7.4. The required packages are as follows:  
- dgl==0.3.1
- numpy==1.18.1
- pandas==0.25.1
- scipy==1.5.2
- torch==1.9.0

It can be installed by the following command.  

```
pip install -r requirement.txt
```

## <span id='dataset'> Dataset </span> 

| Relations(A-B) | Number of A | Number of B | Number of A-B |
|--|--|--|--|
| Drug-disease  |  1482 | 793  |  11540 |
|  Drug-protein |  1482 |  2077 | 11407  |
|  Disease-gene |   793|   6365|  18844 |  

<!-- ## Usage -->
## <span id='usage'> Usage </span> 
1.Create checkpoint/ and  dataset/preprocess_NSAP directories.  

2.run the file /dataset/preprocess_NSAP_NEW.ipynb, and generate all the files we need.  

3.Execute the following command from the home directory:

```
python nsap.py 
```  

## <span id='training'> Training </span> 

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py).  

Some important hyper-parameters are listed here.  
- samples :  It specifies the number of sampled neighboor.
- num_heads : It specifies the number of the attention heads.
