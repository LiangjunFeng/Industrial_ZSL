# Industrial_ZSL
Source code of Industrial_ZSL on TE benchmark dataset.   
The details of model can be found in    
 [L. J. Feng, et al. Fault Description Based Attribute Transfer for Zero-Sample Industrial Fault Diagnosis, TII, 2021.](https://ieeexplore.ieee.org/document/9072621)

#### Fast execution in command line:  
python3 TE_benchmark.py      

#### Results Example:  
 ==========================[test classes][3 ,6, 9]===================================  
 beginning...with feature extraction  
loading data...  
test classes: [3, 6, 9]    
train classes: [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14].  
SPCA extracting feature (takes lots of time)...  
(5760, 400) (5760, 1) (5760, 20)   
(1440, 400) (1440, 1) (1440, 20)  
model training...   
NB  
accuracy:  0.6263888888888889  
rf    
accuracy:  0.5187777777777777 

#### All rights reserved, citing the following papers are required for reference:   
[1] L. J. Feng, et al. Fault Description Based Attribute Transfer for Zero-Sample Industrial Fault Diagnosis, TII, 2021.  
[2] L. J. Feng, et al. Transfer Increment for Generalized Zero-Shot Learning, TNNLS, 2021.  
