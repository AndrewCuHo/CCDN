# CCDN: [Cocktail Causal Container]
This is a PyTorch implementation of Cocktail Causal Container proposed by our paper "End to End Video based Cocktail Causal Container for Blood Pressure Estimation and Glucose Prediction".

## Blood Pressure Estimation and Blood Glucose Prediction

### Main results on PPG-BP

| Model      |  acc_bp@1 | acc_bg@1| Model |
| :---       |     :---: |  :---:  |  :---:  |
| Cocktail Causal Container  |   75.0 |  91.7  | soon |


### Main results on Clinical

| Model      | overall  rmse_bg | acc_bp@1 | acc_bg@1| Model |
| :---       |     :---: |  :---: |  :---:  |  :---:  |
| Cocktail Causal Container  |   0.766 |   89.0 |  89.0  | soon |


**The code and model will be released soon.**

Model summary is shown below (using pytorch-summary package):
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================  
            Conv2d-1           [-1, 32, 50, 50]           5,184  
              SiLU-2           [-1, 32, 50, 50]               0  
    BatchNormAct2d-3           [-1, 32, 50, 50]              64  
         ConvBnAct-4           [-1, 32, 50, 50]               0  
            Conv2d-5           [-1, 32, 50, 50]             288  
          Identity-6           [-1, 32, 50, 50]               0  
    BatchNormAct2d-7           [-1, 32, 50, 50]              64  
         ConvBnAct-8           [-1, 32, 50, 50]               0  
             ReLU6-9           [-1, 32, 50, 50]               0  
           Conv2d-10           [-1, 16, 50, 50]             512  
         Identity-11           [-1, 16, 50, 50]               0  
   BatchNormAct2d-12           [-1, 16, 50, 50]              32  
        ConvBnAct-13           [-1, 16, 50, 50]               0  
 LinearBottleneck-14           [-1, 16, 50, 50]               0  
           Conv2d-15           [-1, 96, 50, 50]           1,536  
             SiLU-16           [-1, 96, 50, 50]               0  
   BatchNormAct2d-17           [-1, 96, 50, 50]             192  
        ConvBnAct-18           [-1, 96, 50, 50]               0  
           Conv2d-19           [-1, 96, 25, 25]             864  
         Identity-20           [-1, 96, 25, 25]               0  
   BatchNormAct2d-21           [-1, 96, 25, 25]             192  
        ConvBnAct-22           [-1, 96, 25, 25]               0  
            ReLU6-23           [-1, 96, 25, 25]               0  
           Conv2d-24           [-1, 27, 25, 25]           2,592  
         Identity-25           [-1, 27, 25, 25]               0  
   BatchNormAct2d-26           [-1, 27, 25, 25]              54  
        ConvBnAct-27           [-1, 27, 25, 25]               0  
 LinearBottleneck-28           [-1, 27, 25, 25]               0  
           Conv2d-29          [-1, 162, 25, 25]           4,374  
             SiLU-30          [-1, 162, 25, 25]               0  
   BatchNormAct2d-31          [-1, 162, 25, 25]             324  
        ConvBnAct-32          [-1, 162, 25, 25]               0  
           Conv2d-33          [-1, 162, 25, 25]           1,458  
         Identity-34          [-1, 162, 25, 25]               0  
   BatchNormAct2d-35          [-1, 162, 25, 25]             324  
        ConvBnAct-36          [-1, 162, 25, 25]               0  
            ReLU6-37          [-1, 162, 25, 25]               0  
           Conv2d-38           [-1, 38, 25, 25]           6,156  
         Identity-39           [-1, 38, 25, 25]               0  
   BatchNormAct2d-40           [-1, 38, 25, 25]              76  
        ConvBnAct-41           [-1, 38, 25, 25]               0  
 LinearBottleneck-42           [-1, 38, 25, 25]               0  
           Conv2d-43          [-1, 228, 25, 25]           8,664  
             SiLU-44          [-1, 228, 25, 25]               0  
   BatchNormAct2d-45          [-1, 228, 25, 25]             456  
        ConvBnAct-46          [-1, 228, 25, 25]               0  
           Conv2d-47          [-1, 228, 13, 13]           2,052  
         Identity-48          [-1, 228, 13, 13]               0  
   BatchNormAct2d-49          [-1, 228, 13, 13]             456  
        ConvBnAct-50          [-1, 228, 13, 13]               0  
           Conv2d-51             [-1, 19, 1, 1]           4,351  
      BatchNorm2d-52             [-1, 19, 1, 1]              38  
             ReLU-53             [-1, 19, 1, 1]               0  
           Conv2d-54            [-1, 228, 1, 1]           4,560  
          Sigmoid-55            [-1, 228, 1, 1]               0  
       SEWithNorm-56          [-1, 228, 13, 13]               0  
            ReLU6-57          [-1, 228, 13, 13]               0  
           Conv2d-58           [-1, 50, 13, 13]          11,400  
         Identity-59           [-1, 50, 13, 13]               0  
   BatchNormAct2d-60           [-1, 50, 13, 13]             100  
        ConvBnAct-61           [-1, 50, 13, 13]               0  
 LinearBottleneck-62           [-1, 50, 13, 13]               0  
           Conv2d-63          [-1, 300, 13, 13]          15,000  
             SiLU-64          [-1, 300, 13, 13]               0  
   BatchNormAct2d-65          [-1, 300, 13, 13]             600  
        ConvBnAct-66          [-1, 300, 13, 13]               0  
           Conv2d-67          [-1, 300, 13, 13]           2,700  
         Identity-68          [-1, 300, 13, 13]               0  
   BatchNormAct2d-69          [-1, 300, 13, 13]             600  
        ConvBnAct-70          [-1, 300, 13, 13]               0  
           Conv2d-71             [-1, 25, 1, 1]           7,525  
      BatchNorm2d-72             [-1, 25, 1, 1]              50  
             ReLU-73             [-1, 25, 1, 1]               0  
           Conv2d-74            [-1, 300, 1, 1]           7,800  
          Sigmoid-75            [-1, 300, 1, 1]               0  
       SEWithNorm-76          [-1, 300, 13, 13]               0  
            ReLU6-77          [-1, 300, 13, 13]               0   
           Conv2d-78           [-1, 61, 13, 13]          18,300  
         Identity-79           [-1, 61, 13, 13]               0  
   BatchNormAct2d-80           [-1, 61, 13, 13]             122  
        ConvBnAct-81           [-1, 61, 13, 13]               0  
 LinearBottleneck-82           [-1, 61, 13, 13]               0  
           Conv2d-83          [-1, 366, 13, 13]          22,326  
             SiLU-84          [-1, 366, 13, 13]               0  
   BatchNormAct2d-85          [-1, 366, 13, 13]             732  
        ConvBnAct-86          [-1, 366, 13, 13]               0  
           Conv2d-87            [-1, 366, 7, 7]           3,294  
         Identity-88            [-1, 366, 7, 7]               0  
   BatchNormAct2d-89            [-1, 366, 7, 7]             732  
        ConvBnAct-90            [-1, 366, 7, 7]               0  
           Conv2d-91             [-1, 30, 1, 1]          11,010  
      BatchNorm2d-92             [-1, 30, 1, 1]              60  
             ReLU-93             [-1, 30, 1, 1]               0  
           Conv2d-94            [-1, 366, 1, 1]          11,346  
          Sigmoid-95            [-1, 366, 1, 1]               0  
       SEWithNorm-96            [-1, 366, 7, 7]               0  
            ReLU6-97            [-1, 366, 7, 7]               0  
           Conv2d-98             [-1, 72, 7, 7]          26,352  
         Identity-99             [-1, 72, 7, 7]               0  
  BatchNormAct2d-100             [-1, 72, 7, 7]             144  
       ConvBnAct-101             [-1, 72, 7, 7]               0  
LinearBottleneck-102             [-1, 72, 7, 7]               0  
          Conv2d-103            [-1, 432, 7, 7]          31,104  
            SiLU-104            [-1, 432, 7, 7]               0  
  BatchNormAct2d-105            [-1, 432, 7, 7]             864  
       ConvBnAct-106            [-1, 432, 7, 7]               0  
          Conv2d-107            [-1, 432, 7, 7]           3,888   
        Identity-108            [-1, 432, 7, 7]               0  
  BatchNormAct2d-109            [-1, 432, 7, 7]             864  
       ConvBnAct-110            [-1, 432, 7, 7]               0  
          Conv2d-111             [-1, 36, 1, 1]          15,588  
     BatchNorm2d-112             [-1, 36, 1, 1]              72  
            ReLU-113             [-1, 36, 1, 1]               0  
          Conv2d-114            [-1, 432, 1, 1]          15,984  
         Sigmoid-115            [-1, 432, 1, 1]               0  
      SEWithNorm-116            [-1, 432, 7, 7]               0  
           ReLU6-117            [-1, 432, 7, 7]               0  
          Conv2d-118             [-1, 84, 7, 7]          36,288  
        Identity-119             [-1, 84, 7, 7]               0  
  BatchNormAct2d-120             [-1, 84, 7, 7]             168  
       ConvBnAct-121             [-1, 84, 7, 7]               0  
LinearBottleneck-122             [-1, 84, 7, 7]               0  
          Conv2d-123            [-1, 504, 7, 7]          42,336  
            SiLU-124            [-1, 504, 7, 7]               0  
  BatchNormAct2d-125            [-1, 504, 7, 7]           1,008  
       ConvBnAct-126            [-1, 504, 7, 7]               0  
          Conv2d-127            [-1, 504, 7, 7]           4,536  
        Identity-128            [-1, 504, 7, 7]               0  
  BatchNormAct2d-129            [-1, 504, 7, 7]           1,008  
       ConvBnAct-130            [-1, 504, 7, 7]               0  
          Conv2d-131             [-1, 42, 1, 1]          21,210  
     BatchNorm2d-132             [-1, 42, 1, 1]              84  
            ReLU-133             [-1, 42, 1, 1]               0
          Conv2d-134            [-1, 504, 1, 1]          21,672  
         Sigmoid-135            [-1, 504, 1, 1]               0  
      SEWithNorm-136            [-1, 504, 7, 7]               0  
           ReLU6-137            [-1, 504, 7, 7]               0  
          Conv2d-138             [-1, 95, 7, 7]          47,880  
        Identity-139             [-1, 95, 7, 7]               0  
  BatchNormAct2d-140             [-1, 95, 7, 7]             190  
       ConvBnAct-141             [-1, 95, 7, 7]               0  
LinearBottleneck-142             [-1, 95, 7, 7]               0  
          Conv2d-143            [-1, 570, 7, 7]          54,150  
            SiLU-144            [-1, 570, 7, 7]               0  
  BatchNormAct2d-145            [-1, 570, 7, 7]           1,140  
       ConvBnAct-146            [-1, 570, 7, 7]               0  
          Conv2d-147            [-1, 570, 7, 7]           5,130  
        Identity-148            [-1, 570, 7, 7]               0  
  BatchNormAct2d-149            [-1, 570, 7, 7]           1,140  
       ConvBnAct-150            [-1, 570, 7, 7]               0  
          Conv2d-151             [-1, 47, 1, 1]          26,837  
     BatchNorm2d-152             [-1, 47, 1, 1]              94  
            ReLU-153             [-1, 47, 1, 1]               0  
          Conv2d-154            [-1, 570, 1, 1]          27,360  
         Sigmoid-155            [-1, 570, 1, 1]               0  
      SEWithNorm-156            [-1, 570, 7, 7]               0  
           ReLU6-157            [-1, 570, 7, 7]               0  
          Conv2d-158            [-1, 106, 7, 7]          60,420  
        Identity-159            [-1, 106, 7, 7]               0  
  BatchNormAct2d-160            [-1, 106, 7, 7]             212  
       ConvBnAct-161            [-1, 106, 7, 7]               0  
LinearBottleneck-162            [-1, 106, 7, 7]               0  
          Conv2d-163            [-1, 636, 7, 7]          67,416  
            SiLU-164            [-1, 636, 7, 7]               0  
  BatchNormAct2d-165            [-1, 636, 7, 7]           1,272  
       ConvBnAct-166            [-1, 636, 7, 7]               0  
          Conv2d-167            [-1, 636, 7, 7]           5,724  
        Identity-168            [-1, 636, 7, 7]               0  
  BatchNormAct2d-169            [-1, 636, 7, 7]           1,272  
       ConvBnAct-170            [-1, 636, 7, 7]               0  
          Conv2d-171             [-1, 53, 1, 1]          33,761  
     BatchNorm2d-172             [-1, 53, 1, 1]             106  
            ReLU-173             [-1, 53, 1, 1]               0  
          Conv2d-174            [-1, 636, 1, 1]          34,344  
         Sigmoid-175            [-1, 636, 1, 1]               0  
      SEWithNorm-176            [-1, 636, 7, 7]               0  
           ReLU6-177            [-1, 636, 7, 7]               0  
          Conv2d-178            [-1, 117, 7, 7]          74,412  
        Identity-179            [-1, 117, 7, 7]               0  
  BatchNormAct2d-180            [-1, 117, 7, 7]             234  
       ConvBnAct-181            [-1, 117, 7, 7]               0  
LinearBottleneck-182            [-1, 117, 7, 7]               0  
          Conv2d-183            [-1, 702, 7, 7]          82,134  
            SiLU-184            [-1, 702, 7, 7]               0  
  BatchNormAct2d-185            [-1, 702, 7, 7]           1,404  
       ConvBnAct-186            [-1, 702, 7, 7]               0  
          Conv2d-187            [-1, 702, 7, 7]           6,318  
        Identity-188            [-1, 702, 7, 7]               0  
  BatchNormAct2d-189            [-1, 702, 7, 7]           1,404  
       ConvBnAct-190            [-1, 702, 7, 7]               0  
          Conv2d-191             [-1, 58, 1, 1]          40,774  
     BatchNorm2d-192             [-1, 58, 1, 1]             116  
            ReLU-193             [-1, 58, 1, 1]               0  
          Conv2d-194            [-1, 702, 1, 1]          41,418  
         Sigmoid-195            [-1, 702, 1, 1]               0  
      SEWithNorm-196            [-1, 702, 7, 7]               0  
           ReLU6-197            [-1, 702, 7, 7]               0  
          Conv2d-198            [-1, 128, 7, 7]          89,856  
        Identity-199            [-1, 128, 7, 7]               0  
  BatchNormAct2d-200            [-1, 128, 7, 7]             256  
 ...
