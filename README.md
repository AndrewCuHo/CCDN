The code will be released after the paper is accepted
Model summary is shown below:
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
       ConvBnAct-201            [-1, 128, 7, 7]               0
LinearBottleneck-202            [-1, 128, 7, 7]               0
          Conv2d-203            [-1, 768, 7, 7]          98,304
            SiLU-204            [-1, 768, 7, 7]               0
  BatchNormAct2d-205            [-1, 768, 7, 7]           1,536
       ConvBnAct-206            [-1, 768, 7, 7]               0
          Conv2d-207            [-1, 768, 4, 4]           6,912
        Identity-208            [-1, 768, 4, 4]               0
  BatchNormAct2d-209            [-1, 768, 4, 4]           1,536
       ConvBnAct-210            [-1, 768, 4, 4]               0
          Conv2d-211             [-1, 64, 1, 1]          49,216
     BatchNorm2d-212             [-1, 64, 1, 1]             128
            ReLU-213             [-1, 64, 1, 1]               0
          Conv2d-214            [-1, 768, 1, 1]          49,920
         Sigmoid-215            [-1, 768, 1, 1]               0
      SEWithNorm-216            [-1, 768, 4, 4]               0
           ReLU6-217            [-1, 768, 4, 4]               0
          Conv2d-218            [-1, 140, 4, 4]         107,520
        Identity-219            [-1, 140, 4, 4]               0
  BatchNormAct2d-220            [-1, 140, 4, 4]             280
       ConvBnAct-221            [-1, 140, 4, 4]               0
LinearBottleneck-222            [-1, 140, 4, 4]               0
          Conv2d-223            [-1, 840, 4, 4]         117,600
            SiLU-224            [-1, 840, 4, 4]               0
  BatchNormAct2d-225            [-1, 840, 4, 4]           1,680
       ConvBnAct-226            [-1, 840, 4, 4]               0
          Conv2d-227            [-1, 840, 4, 4]           7,560
        Identity-228            [-1, 840, 4, 4]               0
  BatchNormAct2d-229            [-1, 840, 4, 4]           1,680
       ConvBnAct-230            [-1, 840, 4, 4]               0
          Conv2d-231             [-1, 70, 1, 1]          58,870
     BatchNorm2d-232             [-1, 70, 1, 1]             140
            ReLU-233             [-1, 70, 1, 1]               0
          Conv2d-234            [-1, 840, 1, 1]          59,640
         Sigmoid-235            [-1, 840, 1, 1]               0
      SEWithNorm-236            [-1, 840, 4, 4]               0
           ReLU6-237            [-1, 840, 4, 4]               0
          Conv2d-238            [-1, 151, 4, 4]         126,840
        Identity-239            [-1, 151, 4, 4]               0
  BatchNormAct2d-240            [-1, 151, 4, 4]             302
       ConvBnAct-241            [-1, 151, 4, 4]               0
LinearBottleneck-242            [-1, 151, 4, 4]               0
          Conv2d-243            [-1, 906, 4, 4]         136,806
            SiLU-244            [-1, 906, 4, 4]               0
  BatchNormAct2d-245            [-1, 906, 4, 4]           1,812
       ConvBnAct-246            [-1, 906, 4, 4]               0
          Conv2d-247            [-1, 906, 4, 4]           8,154
        Identity-248            [-1, 906, 4, 4]               0
  BatchNormAct2d-249            [-1, 906, 4, 4]           1,812
       ConvBnAct-250            [-1, 906, 4, 4]               0
          Conv2d-251             [-1, 75, 1, 1]          68,025
     BatchNorm2d-252             [-1, 75, 1, 1]             150
            ReLU-253             [-1, 75, 1, 1]               0
          Conv2d-254            [-1, 906, 1, 1]          68,856
         Sigmoid-255            [-1, 906, 1, 1]               0
      SEWithNorm-256            [-1, 906, 4, 4]               0
           ReLU6-257            [-1, 906, 4, 4]               0
          Conv2d-258            [-1, 162, 4, 4]         146,772
        Identity-259            [-1, 162, 4, 4]               0
  BatchNormAct2d-260            [-1, 162, 4, 4]             324
       ConvBnAct-261            [-1, 162, 4, 4]               0
LinearBottleneck-262            [-1, 162, 4, 4]               0
          Conv2d-263            [-1, 972, 4, 4]         157,464
            SiLU-264            [-1, 972, 4, 4]               0
  BatchNormAct2d-265            [-1, 972, 4, 4]           1,944
       ConvBnAct-266            [-1, 972, 4, 4]               0
          Conv2d-267            [-1, 972, 4, 4]           8,748
        Identity-268            [-1, 972, 4, 4]               0
  BatchNormAct2d-269            [-1, 972, 4, 4]           1,944
       ConvBnAct-270            [-1, 972, 4, 4]               0
          Conv2d-271             [-1, 81, 1, 1]          78,813
     BatchNorm2d-272             [-1, 81, 1, 1]             162
            ReLU-273             [-1, 81, 1, 1]               0
          Conv2d-274            [-1, 972, 1, 1]          79,704
         Sigmoid-275            [-1, 972, 1, 1]               0
      SEWithNorm-276            [-1, 972, 4, 4]               0
           ReLU6-277            [-1, 972, 4, 4]               0
          Conv2d-278            [-1, 174, 4, 4]         169,128
        Identity-279            [-1, 174, 4, 4]               0
  BatchNormAct2d-280            [-1, 174, 4, 4]             348
       ConvBnAct-281            [-1, 174, 4, 4]               0
LinearBottleneck-282            [-1, 174, 4, 4]               0
          Conv2d-283           [-1, 1044, 4, 4]         181,656
            SiLU-284           [-1, 1044, 4, 4]               0
  BatchNormAct2d-285           [-1, 1044, 4, 4]           2,088
       ConvBnAct-286           [-1, 1044, 4, 4]               0
          Conv2d-287           [-1, 1044, 4, 4]           9,396
        Identity-288           [-1, 1044, 4, 4]               0
  BatchNormAct2d-289           [-1, 1044, 4, 4]           2,088
       ConvBnAct-290           [-1, 1044, 4, 4]               0
          Conv2d-291             [-1, 87, 1, 1]          90,915
     BatchNorm2d-292             [-1, 87, 1, 1]             174
            ReLU-293             [-1, 87, 1, 1]               0
          Conv2d-294           [-1, 1044, 1, 1]          91,872
         Sigmoid-295           [-1, 1044, 1, 1]               0
      SEWithNorm-296           [-1, 1044, 4, 4]               0
           ReLU6-297           [-1, 1044, 4, 4]               0
          Conv2d-298            [-1, 185, 4, 4]         193,140
        Identity-299            [-1, 185, 4, 4]               0
  BatchNormAct2d-300            [-1, 185, 4, 4]             370
       ConvBnAct-301            [-1, 185, 4, 4]               0
LinearBottleneck-302            [-1, 185, 4, 4]               0
          Conv2d-303           [-1, 1280, 4, 4]         236,800
            SiLU-304           [-1, 1280, 4, 4]               0
  BatchNormAct2d-305           [-1, 1280, 4, 4]           2,560
       ConvBnAct-306           [-1, 1280, 4, 4]               0
        Identity-307           [-1, 1280, 4, 4]               0
SelectAdaptivePool2d-308           [-1, 1280, 4, 4]               0
        Identity-309           [-1, 1280, 4, 4]               0
  ClassifierHead-310           [-1, 1280, 4, 4]               0
        ReXNetV1-311           [-1, 1280, 4, 4]               0
     PreEmphasis-312                 [-1, 1800]               0
     Spectrogram-313              [-1, 257, 31]               0
        MelScale-314               [-1, 80, 31]               0
  MelSpectrogram-315               [-1, 80, 31]               0
        FbankAug-316               [-1, 80, 31]               0
     PreEmphasis-317                 [-1, 1800]               0
     Spectrogram-318              [-1, 257, 31]               0
        MelScale-319               [-1, 80, 31]               0
  MelSpectrogram-320               [-1, 80, 31]               0
        FbankAug-321               [-1, 80, 31]               0
     PreEmphasis-322                 [-1, 1800]               0
     Spectrogram-323              [-1, 257, 31]               0
        MelScale-324               [-1, 80, 31]               0
  MelSpectrogram-325               [-1, 80, 31]               0
        FbankAug-326               [-1, 80, 31]               0
     PreEmphasis-327                 [-1, 1800]               0
     Spectrogram-328              [-1, 257, 31]               0
        MelScale-329               [-1, 80, 31]               0
  MelSpectrogram-330               [-1, 80, 31]               0
        FbankAug-331               [-1, 80, 31]               0
     PreEmphasis-332                 [-1, 1800]               0
     Spectrogram-333              [-1, 257, 31]               0
        MelScale-334               [-1, 80, 31]               0
  MelSpectrogram-335               [-1, 80, 31]               0
        FbankAug-336               [-1, 80, 31]               0
     PreEmphasis-337                 [-1, 1800]               0
     Spectrogram-338              [-1, 257, 31]               0
        MelScale-339               [-1, 80, 31]               0
  MelSpectrogram-340               [-1, 80, 31]               0
        FbankAug-341               [-1, 80, 31]               0
          Conv1d-342              [-1, 512, 31]       1,228,800
     BatchNorm1d-343              [-1, 512, 31]           1,024
    Conv1dReluBn-344              [-1, 512, 31]               0
          Conv1d-345              [-1, 512, 31]         262,144
     BatchNorm1d-346              [-1, 512, 31]           1,024
    Conv1dReluBn-347              [-1, 512, 31]               0
          Conv1d-348               [-1, 64, 31]          12,288
     BatchNorm1d-349               [-1, 64, 31]             128
          Conv1d-350               [-1, 64, 31]          12,288
     BatchNorm1d-351               [-1, 64, 31]             128
          Conv1d-352               [-1, 64, 31]          12,288
     BatchNorm1d-353               [-1, 64, 31]             128
          Conv1d-354               [-1, 64, 31]          12,288
     BatchNorm1d-355               [-1, 64, 31]             128
          Conv1d-356               [-1, 64, 31]          12,288
     BatchNorm1d-357               [-1, 64, 31]             128
          Conv1d-358               [-1, 64, 31]          12,288
     BatchNorm1d-359               [-1, 64, 31]             128
          Conv1d-360               [-1, 64, 31]          12,288
     BatchNorm1d-361               [-1, 64, 31]             128
Res2Conv1dReluBn-362              [-1, 512, 31]               0
          Conv1d-363              [-1, 512, 31]         262,144
     BatchNorm1d-364              [-1, 512, 31]           1,024
    Conv1dReluBn-365              [-1, 512, 31]               0
          Linear-366                  [-1, 256]         131,328
          Linear-367                  [-1, 512]         131,584
      SE_Connect-368              [-1, 512, 31]               0
          Conv1d-369              [-1, 512, 31]         262,144
     BatchNorm1d-370              [-1, 512, 31]           1,024
    Conv1dReluBn-371              [-1, 512, 31]               0
          Conv1d-372               [-1, 64, 31]          12,288
     BatchNorm1d-373               [-1, 64, 31]             128
          Conv1d-374               [-1, 64, 31]          12,288
     BatchNorm1d-375               [-1, 64, 31]             128
          Conv1d-376               [-1, 64, 31]          12,288
     BatchNorm1d-377               [-1, 64, 31]             128
          Conv1d-378               [-1, 64, 31]          12,288
     BatchNorm1d-379               [-1, 64, 31]             128
          Conv1d-380               [-1, 64, 31]          12,288
     BatchNorm1d-381               [-1, 64, 31]             128
          Conv1d-382               [-1, 64, 31]          12,288
     BatchNorm1d-383               [-1, 64, 31]             128
          Conv1d-384               [-1, 64, 31]          12,288
     BatchNorm1d-385               [-1, 64, 31]             128
Res2Conv1dReluBn-386              [-1, 512, 31]               0
          Conv1d-387              [-1, 512, 31]         262,144
     BatchNorm1d-388              [-1, 512, 31]           1,024
    Conv1dReluBn-389              [-1, 512, 31]               0
          Linear-390                  [-1, 256]         131,328
          Linear-391                  [-1, 512]         131,584
      SE_Connect-392              [-1, 512, 31]               0
          Conv1d-393              [-1, 512, 31]         262,144
     BatchNorm1d-394              [-1, 512, 31]           1,024
    Conv1dReluBn-395              [-1, 512, 31]               0
          Conv1d-396               [-1, 64, 31]          12,288
     BatchNorm1d-397               [-1, 64, 31]             128
          Conv1d-398               [-1, 64, 31]          12,288
     BatchNorm1d-399               [-1, 64, 31]             128
          Conv1d-400               [-1, 64, 31]          12,288
     BatchNorm1d-401               [-1, 64, 31]             128
          Conv1d-402               [-1, 64, 31]          12,288
     BatchNorm1d-403               [-1, 64, 31]             128
          Conv1d-404               [-1, 64, 31]          12,288
     BatchNorm1d-405               [-1, 64, 31]             128
          Conv1d-406               [-1, 64, 31]          12,288
     BatchNorm1d-407               [-1, 64, 31]             128
          Conv1d-408               [-1, 64, 31]          12,288
     BatchNorm1d-409               [-1, 64, 31]             128
Res2Conv1dReluBn-410              [-1, 512, 31]               0
          Conv1d-411              [-1, 512, 31]         262,144
     BatchNorm1d-412              [-1, 512, 31]           1,024
    Conv1dReluBn-413              [-1, 512, 31]               0
          Linear-414                  [-1, 256]         131,328
          Linear-415                  [-1, 512]         131,584
      SE_Connect-416              [-1, 512, 31]               0
          Conv1d-417             [-1, 1536, 31]       2,360,832
ECAPA_TDNN_Encode-418             [-1, 1536, 31]               0
          Conv1d-419             [-1, 1536, 31]       2,360,832
     BatchNorm1d-420             [-1, 1536, 31]           3,072
            ReLU-421             [-1, 1536, 31]               0
          Conv1d-422              [-1, 128, 31]         196,736
          Conv1d-423             [-1, 1536, 31]         198,144
AttentiveStatsPool-424                 [-1, 3072]               0
     BatchNorm1d-425                 [-1, 3072]           6,144
          Linear-426                  [-1, 192]         590,016
     BatchNorm1d-427                  [-1, 192]             384
ECAPA_TDNN_Dencode-428                  [-1, 192]               0
          Linear-429                  [-1, 100]          19,300
     BatchNorm1d-430                  [-1, 100]             200
            ReLU-431                  [-1, 100]               0
         Dropout-432                  [-1, 100]               0
          Linear-433                   [-1, 64]           6,464
     BatchNorm1d-434                   [-1, 64]             128
            ReLU-435                   [-1, 64]               0
          Linear-436                   [-1, 61]           3,965
          Conv2d-437           [-1, 1280, 4, 4]          11,520
          Conv2d-438           [-1, 3840, 4, 4]       4,915,200
AdaptiveAvgPool2d-439           [-1, 1280, 1, 1]               0
          Conv1d-440              [-1, 1, 1280]               3
         Sigmoid-441           [-1, 1280, 1, 1]               0
          Conv2d-442           [-1, 1280, 4, 4]       3,276,800
     BatchNorm2d-443           [-1, 1280, 4, 4]           2,560
            ReLU-444           [-1, 1280, 4, 4]               0
 MHConvAttention-445           [-1, 1280, 4, 4]               0
AdaptiveAvgPool2d-446           [-1, 1280, 1, 1]               0
         Dropout-447           [-1, 1280, 1, 1]               0
          Linear-448                  [-1, 380]         486,780
     BatchNorm1d-449                  [-1, 380]             760
            ReLU-450                  [-1, 380]               0
         Dropout-451                  [-1, 380]               0
          Linear-452                  [-1, 100]          38,100
     BatchNorm1d-453                  [-1, 100]             200
            ReLU-454                  [-1, 100]               0
          Linear-455                   [-1, 61]           6,161
          Conv1d-456             [-1, 1536, 31]       2,360,832
     BatchNorm1d-457             [-1, 1536, 31]           3,072
            ReLU-458             [-1, 1536, 31]               0
          Conv1d-459              [-1, 128, 31]         196,736
          Conv1d-460             [-1, 1536, 31]         198,144
AttentiveStatsPool-461                 [-1, 3072]               0
     BatchNorm1d-462                 [-1, 3072]           6,144
          Linear-463                  [-1, 192]         590,016
     BatchNorm1d-464                  [-1, 192]             384
ECAPA_TDNN_Dencode-465                  [-1, 192]               0
          Linear-466                  [-1, 100]          19,300
     BatchNorm1d-467                  [-1, 100]             200
            ReLU-468                  [-1, 100]               0
         Dropout-469                  [-1, 100]               0
          Linear-470                   [-1, 64]           6,464
     BatchNorm1d-471                   [-1, 64]             128
            ReLU-472                   [-1, 64]               0
          Conv2d-473           [-1, 1280, 4, 4]          11,520
          Conv2d-474           [-1, 3840, 4, 4]       4,915,200
AdaptiveAvgPool2d-475           [-1, 1280, 1, 1]               0
          Conv1d-476              [-1, 1, 1280]               3
         Sigmoid-477           [-1, 1280, 1, 1]               0
          Conv2d-478           [-1, 1280, 4, 4]       3,276,800
     BatchNorm2d-479           [-1, 1280, 4, 4]           2,560
            ReLU-480           [-1, 1280, 4, 4]               0
 MHConvAttention-481           [-1, 1280, 4, 4]               0
AdaptiveAvgPool2d-482           [-1, 1280, 1, 1]               0
         Dropout-483           [-1, 1280, 1, 1]               0
          Linear-484                  [-1, 380]         486,780
     BatchNorm1d-485                  [-1, 380]             760
            ReLU-486                  [-1, 380]               0
         Dropout-487                  [-1, 380]               0
          Linear-488                  [-1, 100]          38,100
     BatchNorm1d-489                  [-1, 100]             200
            ReLU-490                  [-1, 100]               0
          Conv1d-491              [-1, 512, 31]         786,944
     BatchNorm1d-492              [-1, 512, 31]           1,024
            ReLU-493              [-1, 512, 31]               0
          Conv1d-494              [-1, 128, 31]          65,664
     BatchNorm1d-495              [-1, 128, 31]             256
            ReLU-496              [-1, 128, 31]               0
          Conv1d-497                [-1, 1, 31]             129
     BatchNorm1d-498                [-1, 1, 31]               2
      Subtractor-499                [-1, 1, 31]               0
          Conv1d-500              [-1, 512, 31]         786,944
     BatchNorm1d-501              [-1, 512, 31]           1,024
            ReLU-502              [-1, 512, 31]               0
          Conv1d-503              [-1, 128, 31]          65,664
     BatchNorm1d-504              [-1, 128, 31]             256
            ReLU-505              [-1, 128, 31]               0
          Conv1d-506                [-1, 1, 31]             129
     BatchNorm1d-507                [-1, 1, 31]               2
      Subtractor-508                [-1, 1, 31]               0
          Conv2d-509            [-1, 192, 4, 4]         245,952
     BatchNorm2d-510            [-1, 192, 4, 4]             384
            ReLU-511            [-1, 192, 4, 4]               0
AdaptiveMaxPool2d-512            [-1, 192, 1, 1]               0
       LayerNorm-513               [-1, 2, 192]             384
       LayerNorm-514               [-1, 2, 192]             384
          Linear-515               [-1, 2, 768]         148,224
            GELU-516               [-1, 2, 768]               0
          Linear-517               [-1, 2, 192]         147,648
         Dropout-518               [-1, 2, 192]               0
     FeedForward-519               [-1, 2, 192]               0
       LayerNorm-520               [-1, 2, 192]             384
    EncoderBlock-521               [-1, 2, 192]               0
       LayerNorm-522               [-1, 2, 192]             384
          Linear-523               [-1, 2, 768]         148,224
            GELU-524               [-1, 2, 768]               0
          Linear-525               [-1, 2, 192]         147,648
         Dropout-526               [-1, 2, 192]               0
     FeedForward-527               [-1, 2, 192]               0
       LayerNorm-528               [-1, 2, 192]             384
    EncoderBlock-529               [-1, 2, 192]               0
          Linear-530               [-1, 2, 192]          37,056
            Fnet-531               [-1, 2, 192]               0
MultiheadAttention-532  [[-1, 2, 192], [-1, 2, 2]]               0
         Dropout-533               [-1, 2, 192]               0
       LayerNorm-534               [-1, 2, 192]             384
          Linear-535              [-1, 2, 2048]         395,264
         Dropout-536              [-1, 2, 2048]               0
          Linear-537               [-1, 2, 192]         393,408
         Dropout-538               [-1, 2, 192]               0
       LayerNorm-539               [-1, 2, 192]             384
TransformerEncoderLayer-540               [-1, 2, 192]               0
MultiheadAttention-541  [[-1, 2, 192], [-1, 2, 2]]               0
         Dropout-542               [-1, 2, 192]               0
       LayerNorm-543               [-1, 2, 192]             384
          Linear-544              [-1, 2, 2048]         395,264
         Dropout-545              [-1, 2, 2048]               0
          Linear-546               [-1, 2, 192]         393,408
         Dropout-547               [-1, 2, 192]               0
       LayerNorm-548               [-1, 2, 192]             384
TransformerEncoderLayer-549               [-1, 2, 192]               0
MultiheadAttention-550  [[-1, 2, 192], [-1, 2, 2]]               0
         Dropout-551               [-1, 2, 192]               0
       LayerNorm-552               [-1, 2, 192]             384
          Linear-553              [-1, 2, 2048]         395,264
         Dropout-554              [-1, 2, 2048]               0
          Linear-555               [-1, 2, 192]         393,408
         Dropout-556               [-1, 2, 192]               0
       LayerNorm-557               [-1, 2, 192]             384
TransformerEncoderLayer-558               [-1, 2, 192]               0
TransformerEncoder-559               [-1, 2, 192]               0
          Linear-560                  [-1, 192]          37,056
     BatchNorm1d-561                  [-1, 192]             384
            ReLU-562                  [-1, 192]               0
         Dropout-563                  [-1, 192]               0
          Linear-564                  [-1, 100]          19,300
     BatchNorm1d-565                  [-1, 100]             200
            ReLU-566                  [-1, 100]               0
          Linear-567                   [-1, 61]           6,161
          Linear-568                  [-1, 100]          19,300
     BatchNorm1d-569                  [-1, 100]             200
            ReLU-570                  [-1, 100]               0
         Dropout-571                  [-1, 100]               0
          Linear-572                   [-1, 61]           6,161
          Conv2d-573            [-1, 192, 4, 4]         245,952
     BatchNorm2d-574            [-1, 192, 4, 4]             384
            ReLU-575            [-1, 192, 4, 4]               0
AdaptiveMaxPool2d-576            [-1, 192, 1, 1]               0
       LayerNorm-577               [-1, 2, 192]             384
       LayerNorm-578               [-1, 2, 192]             384
          Linear-579               [-1, 2, 768]         148,224
            GELU-580               [-1, 2, 768]               0
          Linear-581               [-1, 2, 192]         147,648
         Dropout-582               [-1, 2, 192]               0
     FeedForward-583               [-1, 2, 192]               0
       LayerNorm-584               [-1, 2, 192]             384
    EncoderBlock-585               [-1, 2, 192]               0
       LayerNorm-586               [-1, 2, 192]             384
          Linear-587               [-1, 2, 768]         148,224
            GELU-588               [-1, 2, 768]               0
          Linear-589               [-1, 2, 192]         147,648
         Dropout-590               [-1, 2, 192]               0
     FeedForward-591               [-1, 2, 192]               0
       LayerNorm-592               [-1, 2, 192]             384
    EncoderBlock-593               [-1, 2, 192]               0
          Linear-594               [-1, 2, 192]          37,056
            Fnet-595               [-1, 2, 192]               0
MultiheadAttention-596  [[-1, 2, 192], [-1, 2, 2]]               0
         Dropout-597               [-1, 2, 192]               0
       LayerNorm-598               [-1, 2, 192]             384
          Linear-599              [-1, 2, 2048]         395,264
         Dropout-600              [-1, 2, 2048]               0
          Linear-601               [-1, 2, 192]         393,408
         Dropout-602               [-1, 2, 192]               0
       LayerNorm-603               [-1, 2, 192]             384
TransformerEncoderLayer-604               [-1, 2, 192]               0
MultiheadAttention-605  [[-1, 2, 192], [-1, 2, 2]]               0
         Dropout-606               [-1, 2, 192]               0
       LayerNorm-607               [-1, 2, 192]             384
          Linear-608              [-1, 2, 2048]         395,264
         Dropout-609              [-1, 2, 2048]               0
          Linear-610               [-1, 2, 192]         393,408
         Dropout-611               [-1, 2, 192]               0
       LayerNorm-612               [-1, 2, 192]             384
TransformerEncoderLayer-613               [-1, 2, 192]               0
MultiheadAttention-614  [[-1, 2, 192], [-1, 2, 2]]               0
         Dropout-615               [-1, 2, 192]               0
       LayerNorm-616               [-1, 2, 192]             384
          Linear-617              [-1, 2, 2048]         395,264
         Dropout-618              [-1, 2, 2048]               0
          Linear-619               [-1, 2, 192]         393,408
         Dropout-620               [-1, 2, 192]               0
       LayerNorm-621               [-1, 2, 192]             384
TransformerEncoderLayer-622               [-1, 2, 192]               0
TransformerEncoder-623               [-1, 2, 192]               0
          Linear-624                  [-1, 192]          37,056
     BatchNorm1d-625                  [-1, 192]             384
            ReLU-626                  [-1, 192]               0
         Dropout-627                  [-1, 192]               0
          Linear-628                  [-1, 100]          19,300
     BatchNorm1d-629                  [-1, 100]             200
            ReLU-630                  [-1, 100]               0
          Linear-631                    [-1, 2]             202
          Linear-632                  [-1, 100]          19,300
     BatchNorm1d-633                  [-1, 100]             200
            ReLU-634                  [-1, 100]               0
         Dropout-635                  [-1, 100]               0
          Linear-636                    [-1, 2]             202
