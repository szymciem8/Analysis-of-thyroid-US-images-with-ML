# Anlysis of thyroid US images with ML

Deep learning machine models are employed for the segmentation of tumors and pathological changes in medical images. This paper presents the results of an analysis, based on selected metrics, regarding the U-Net, U2-Net, U-Net 3+, and TransUnet models. Furthermore, it discusses how the selected models address the issue of data heterogeneity.

Trained models can be downloaded from [OneDrive](https://polslpl-my.sharepoint.com/:f:/g/personal/szymcie806_student_polsl_pl/EqeQThhS8S5LotZipdUZqagBcxockNVqYzDHuLMjRVLPXw?e=LPiMM2). You have to be member of the Silesian University of Science organization in order to access those models. 


# Model Evaluation
    
## 1. Classic U-Net

## 1.1. Samsung on Samsung

![png](report_images/output_13_1.png)
    
![png](report_images/output_13_2.png)

    Dice coefficient=0.8680596947669983
    accuracy=0.97763427734375, precision=0.8391287360812659, recall=0.8985059016883311, f1=0.8678028225922826
    loss=0.20392586290836334
    
![png](report_images/output_13_4.png)

## 1.2. GE on GE

![png](report_images/output_15_1.png)
    
![png](report_images/output_15_2.png)
    
    Dice coefficient=0.6949803829193115
    accuracy=0.967244873046875, precision=0.5880118030510502, recall=0.8478587319243605, f1=0.6944232499345185
    loss=0.3548142611980438
    
![png](report_images/output_15_4.png)

## 1.3. Samsung on GE
    
![png](report_images/output_17_1.png)

    Dice coefficient=0.11393407732248306
    accuracy=0.6583203125, precision=0.061414343242882606, recall=0.762953202703277, f1=0.11367810414054286
    loss=0.8677613139152527
    
![png](report_images/output_17_3.png)

## 1.4. GE on Samsung
    
![png](report_images/output_19_1.png)
    
    Dice coefficient=0.5466128587722778
    accuracy=0.94544921875, precision=0.5793330890436057, recall=0.5163787969029184, f1=0.546047418784665
    loss=0.5640836358070374
    
![png](report_images/output_19_3.png)

## 1.5. Mix on Samsung

![png](report_images/output_22_1.png)
    
![png](report_images/output_22_2.png)
    
    Dice coefficient=0.719916582107544
    accuracy=0.93884765625, precision=0.575365770670296, recall=0.9600776931122068, f1=0.7195261236647034
    loss=0.2991807758808136
    
![png](report_images/output_22_4.png)
    

## 1.6. Mix on GE

# 2. U^2-Net

## 2.1. Samsung

![png](report_images/output_28_1.png)
    
![png](report_images/output_28_2.png)
    
    Dice coefficient=0.864187479019165
    accuracy=0.982620849609375, precision=0.913004679939716, recall=0.823764112289993, f1=0.8660916675288518
    loss=0.24412938952445984
    
![png](report_images/output_28_4.png)
    
![png](report_images/output_29_1.png)
    
![png](report_images/output_29_2.png)

    Dice coefficient=0.8476685881614685
    accuracy=0.975244140625, precision=0.7784213850847367, recall=0.9385380713950059, f1=0.8510138113429327
    loss=0.20272256433963776
    
![png](report_images/output_29_4.png)

## 2.2 GE

![png](report_images/output_31_1.png)
    
![png](report_images/output_31_2.png)

    Dice coefficient=0.6172356605529785
    accuracy=0.911033935546875, precision=0.7578125997726779, recall=0.5239369216641944, f1=0.6195375837209424
    loss=0.5250241756439209
    
![png](report_images/output_31_4.png)

## 2.3. Samsung on GE

![png](report_images/output_33_1.png)

    Dice coefficient=0.3906748294830322
    accuracy=0.605242919921875, precision=0.24764432488260654, recall=0.9103535353535354, f1=0.3893683993874518
    loss=0.5890365242958069

![png](report_images/output_33_3.png)
    
## 2.4. GE on Samsung

![png](report_images/output_35_1.png)
    
    Dice coefficient=0.780373215675354
    accuracy=0.97041259765625, precision=0.7482614507368741, recall=0.8253177313700241, f1=0.7849029143445387
    loss=0.30533477663993835
    
![png](report_images/output_35_3.png)
    
## 2.5. Mix on Samsung
    
![png](report_images/output_38_1.png)
    
![png](report_images/output_38_2.png)
    
    Dice coefficient=0.7622494697570801
    accuracy=0.96321533203125, precision=0.802220908075432, recall=0.7296578514866279, f1=0.7642207721077259
    loss=0.3564944863319397
    
![png](report_images/output_38_4.png)
    
# 3. Unet 3+
## 3.1. Samsung

![png](report_images/output_42_1.png)
    
![png](report_images/output_42_2.png)
    
    Dice coefficient=0.6439123153686523
    accuracy=0.96372802734375, precision=0.7504122069138977, recall=0.5637031594415871, f1=0.6437939053921216
    loss=0.4945428669452667

![png](report_images/output_42_4.png)
    
## 3.2. GE
    
![png](report_images/output_44_1.png)
    
![png](report_images/output_44_2.png)
    
    Dice coefficient=0.8474569320678711
    accuracy=0.972708740234375, precision=0.840134395491248, recall=0.8545246592897794, f1=0.847268429645924
    loss=0.24080750346183777
    
![png](report_images/output_44_4.png)
    
## 3.3. Samsung on GE
![png](report_images/output_46_1.png)

    Dice coefficient=0.34440407156944275
    accuracy=0.572071533203125, precision=0.2185278230484012, recall=0.8133608815426997, f1=0.34449835918436034
    loss=0.6409668922424316
    
![png](report_images/output_46_3.png)
    


## 3.4. GE on Samsung

![png](report_images/output_48_1.png)

    Dice coefficient=0.6187209486961365
    accuracy=0.945159912109375, precision=0.6041919653996507, recall=0.6321973125391631, f1=0.6178774656153513
    loss=0.4801424741744995

![png](report_images/output_48_3.png)

## Mix

![png](report_images/output_50_1.png)
    
![png](report_images/output_50_2.png)
    
    Dice coefficient=0.6516767144203186
    accuracy=0.9642138671875, precision=0.49045076680521715, recall=0.9690265486725663, f1=0.6512739990007851
    loss=0.3540329337120056

![png](report_images/output_50_4.png)
    
# 4. TransUnet

## 4.1. Samsung

![png](report_images/output_54_1.png)
    
![png](report_images/output_54_2.png)

    Dice coefficient=0.827840268611908
    accuracy=0.9782820129394532, precision=0.7967899739098024, recall=0.8612665966666918, f1=0.827774638411564
    loss=0.25191575288772583

![png](report_images/output_54_4.png)
    
![png](report_images/output_55_1.png)
    
![png](report_images/output_55_2.png)
    
    Dice coefficient=0.8332201838493347
    accuracy=0.979171142578125, precision=0.8093630638265548, recall=0.8584942955056492, f1=0.8332050332724824
    loss=0.2492772340774536

![png](report_images/output_55_4.png)
    
## 4.2. GE

![png](report_images/output_57_1.png)
    
![png](report_images/output_57_2.png)

    Dice coefficient=0.8399377465248108
    accuracy=0.971876220703125, precision=0.8466739459096254, recall=0.8334550565668536, f1=0.8400124995659872
    loss=0.2563346028327942

![png](report_images/output_57_4.png)
    
## 4.3. Samsung on GE

![png](report_images/output_59_1.png)
    
    Dice coefficient=0.4845745861530304
    accuracy=0.780294189453125, precision=0.35855671763912855, recall=0.7467683831320195, f1=0.48448880805419114
    loss=0.5368759036064148

![png](report_images/output_59_3.png)

## 4.4 GE on Samsung
    
![png](report_images/output_61_1.png)

    Dice coefficient=0.6104358434677124
    accuracy=0.944107666015625, precision=0.4605149828381905, recall=0.9036171930178589, f1=0.6101010789130823
    loss=0.4041687846183777
    
![png](report_images/output_61_3.png)

## 4.5. Mix on Samsung

![png](report_images/output_64_1.png)
    
![png](report_images/output_64_2.png)

    Dice coefficient=0.8459200859069824
    accuracy=0.973375244140625, precision=0.8036803209175226, recall=0.892021515015688, f1=0.8455497567573308
    loss=0.22421154379844666

![png](report_images/output_64_4.png)

# 5. Compare metrics

## 5.1. Samsung on Samsung
    
| Metryki            | U-Net             | U$^2$-Net         | U-Net 3+          | TransUnet         |
|--------------------|-------------------|-------------------|-------------------|-------------------|
| Focal Tversky      | 0.278 +- 2.03e-02 | 0.266 +- 7.24e-03 | 0.305 +- 6.75e-03 | 0.316 +- 2.39e-02 |
| Dokładność         | 0.972 +- 3.73e-03 | 0.973 +- 1.90e-03 | 0.969 +- 1.94e-03 | 0.965 +- 3.73e-03 |
| Średnia dokładność | 0.86 +- 1.64e-02  | 0.864 +- 1.10e-02 | 0.852 +- 1.20e-02 | 0.833 +- 1.75e-02 |
| Precyzja           | 0.729 +- 3.22e-02 | 0.736 +- 2.22e-02 | 0.715 +- 2.54e-02 | 0.677 +- 3.47e-02 |
| Czułość            | 0.864 +- 1.03e-02 | 0.881 +- 5.84e-03 | 0.839 +- 2.32e-02 | 0.843 +- 2.53e-02 |
| F1/Dice            | 0.79 +- 2.29e-02  | 0.801 +- 1.12e-02 | 0.769 +- 6.76e-03 | 0.749 +- 2.43e-02 |
| IoU                | 0.655 +- 3.01e-02 | 0.669 +- 1.58e-02 | 0.625 +- 8.81e-03 | 0.601 +- 3.21e-02 |
| ROC AUC            | 0.946 +- 6.29e-03 | 0.988 +- 6.18e-04 | 0.932 +- 6.76e-03 | 0.96 +- 5.72e-03  |
    

![png](report_images/output_74_0.png)

![png](report_images/output_75_0.png)
    
![png](report_images/output_76_0.png)
    
![png](report_images/output_77_0.png)
    


## 5.2. GE on GE
    
| Metryki            | U-Net             | U$^2$-Net         | U-Net 3+          | TransUnet         |
|--------------------|-------------------|-------------------|-------------------|-------------------|
| Focal Tversky      | 0.366 +- 1.40e-02 | 0.38 +- 7.26e-03  | 0.359 +- 3.30e-03 | 0.344 +- 1.22e-02 |
| Dokładność         | 0.945 +- 3.13e-03 | 0.947 +- 4.21e-03 | 0.947 +- 2.08e-03 | 0.945 +- 6.15e-03 |
| Średnia dokładność | 0.8 +- 9.00e-03   | 0.812 +- 1.43e-02 | 0.809 +- 8.16e-03 | 0.805 +- 1.80e-02 |
| Precyzja           | 0.616 +- 1.74e-02 | 0.644 +- 2.93e-02 | 0.634 +- 1.76e-02 | 0.625 +- 3.75e-02 |
| Czułość            | 0.807 +- 1.42e-02 | 0.774 +- 1.09e-02 | 0.809 +- 1.56e-02 | 0.845 +- 2.48e-02 |
| F1/Dice            | 0.699 +- 1.43e-02 | 0.7 +- 1.37e-02   | 0.709 +- 4.71e-03 | 0.713 +- 1.86e-02 |
| IoU                | 0.538 +- 1.68e-02 | 0.54 +- 1.62e-02  | 0.55 +- 5.69e-03  | 0.555 +- 2.20e-02 |
| ROC AUC            | 0.895 +- 8.19e-03 | 0.961 +- 1.60e-03 | 0.907 +- 5.08e-03 | 0.938 +- 3.22e-03 |

    
![png](report_images/output_82_0.png)
       
![png](report_images/output_83_0.png)
    
![png](report_images/output_84_0.png)
    
![png](report_images/output_85_0.png)
    
## 5.3. Samsung on GE
    
| Metryki            | U-Net             | U$^2$-Net         | U-Net 3+          | TransUnet         |
|--------------------|-------------------|-------------------|-------------------|-------------------|
| Focal Tversky      | 0.572 +- 3.35e-02 | 0.67 +- 1.99e-02  | 0.641 +- 6.64e-02 | 0.615 +- 2.55e-02 |
| Dokładność         | 0.802 +- 3.52e-02 | 0.646 +- 3.93e-02 | 0.663 +- 7.46e-02 | 0.734 +- 3.98e-02 |
| Średnia dokładność | 0.638 +- 2.12e-02 | 0.587 +- 7.22e-03 | 0.61 +- 3.72e-02  | 0.609 +- 1.09e-02 |
| Precyzja           | 0.291 +- 4.34e-02 | 0.181 +- 1.49e-02 | 0.234 +- 7.45e-02 | 0.229 +- 2.26e-02 |
| Czułość            | 0.847 +- 2.27e-02 | 0.94 +- 8.59e-03  | 0.894 +- 2.72e-02 | 0.912 +- 1.28e-02 |
| F1/Dice            | 0.424 +- 4.34e-02 | 0.303 +- 2.07e-02 | 0.346 +- 7.94e-02 | 0.363 +- 2.86e-02 |
| IoU                | 0.273 +- 3.57e-02 | 0.179 +- 1.43e-02 | 0.222 +- 6.59e-02 | 0.223 +- 2.10e-02 |
| ROC AUC            | 0.835 +- 1.56e-02 | 0.908 +- 4.07e-03 | 0.839 +- 2.54e-02 | 0.825 +- 1.57e-02 |
    

![png](report_images/output_89_0.png)
    
![png](report_images/output_90_0.png)
    
![png](report_images/output_91_0.png)
    
![png](report_images/output_92_0.png)
    

## 5.4. GE on Samsung

| Metryki            | U-Net             | U$^2$-Net         | U-Net 3+          | TransUnet         |
|--------------------|-------------------|-------------------|-------------------|-------------------|
| Focal Tversky      | 0.391 +- 2.27e-02 | 0.434 +- 1.91e-02 | 0.398 +- 2.74e-02 | 0.381 +- 1.48e-02 |
| Dokładność         | 0.965 +- 3.02e-03 | 0.96 +- 3.40e-03  | 0.967 +- 1.26e-03 | 0.962 +- 3.99e-03 |
| Średnia dokładność | 0.848 +- 1.47e-02 | 0.827 +- 1.81e-02 | 0.867 +- 1.28e-02 | 0.832 +- 2.12e-02 |
| Precyzja           | 0.715 +- 2.89e-02 | 0.675 +- 3.58e-02 | 0.754 +- 2.75e-02 | 0.679 +- 4.39e-02 |
| Czułość            | 0.715 +- 2.41e-02 | 0.677 +- 2.02e-02 | 0.694 +- 4.43e-02 | 0.754 +- 2.96e-02 |
| F1/Dice            | 0.713 +- 2.22e-02 | 0.674 +- 2.22e-02 | 0.717 +- 1.49e-02 | 0.708 +- 1.72e-02 |
| IoU                | 0.556 +- 2.72e-02 | 0.51 +- 2.62e-02  | 0.559 +- 1.85e-02 | 0.548 +- 2.03e-02 |
| ROC AUC            | 0.899 +- 1.81e-02 | 0.959 +- 7.84e-03 | 0.866 +- 2.16e-02 | 0.945 +- 4.32e-03 |

    
![png](report_images/output_96_0.png)
    
![png](report_images/output_97_0.png)

    
![png](report_images/output_98_0.png)
    
![png](report_images/output_99_0.png)
    


## 5.5. Mix on Samsung

    
| Metryki            | U-Net             | U$^2$-Net         | U-Net 3+          | TransUnet         |
|--------------------|-------------------|-------------------|-------------------|-------------------|
| Focal Tversky      | 0.316 +- 2.75e-02 | 0.256 +- 1.41e-02 | 0.278 +- 6.06e-03 | 0.254 +- 8.75e-03 |
| Dokładność         | 0.959 +- 7.47e-03 | 0.976 +- 1.10e-03 | 0.97 +- 1.27e-03  | 0.975 +- 1.43e-03 |
| Średnia dokładność | 0.812 +- 2.71e-02 | 0.879 +- 5.15e-03 | 0.849 +- 7.37e-03 | 0.868 +- 7.17e-03 |
| Precyzja           | 0.633 +- 5.42e-02 | 0.767 +- 1.05e-02 | 0.707 +- 1.54e-02 | 0.744 +- 1.43e-02 |
| Czułość            | 0.882 +- 1.01e-02 | 0.875 +- 1.83e-02 | 0.88 +- 1.38e-02  | 0.888 +- 7.89e-03 |
| F1/Dice            | 0.731 +- 3.63e-02 | 0.817 +- 9.07e-03 | 0.783 +- 5.91e-03 | 0.809 +- 9.08e-03 |
| IoU                | 0.581 +- 4.47e-02 | 0.691 +- 1.31e-02 | 0.644 +- 7.90e-03 | 0.68 +- 1.30e-02  |
| ROC AUC            | 0.932 +- 4.91e-03 | 0.989 +- 1.19e-03 | 0.942 +- 1.24e-02 | 0.97 +- 2.75e-03  |

    
![png](report_images/output_104_0.png)

![png](report_images/output_105_0.png)
    

## 5.5. Mix on GE
    
| Metryki            | U-Net             | U$^2$-Net         | U-Net 3+          | TransUnet         |
|--------------------|-------------------|-------------------|-------------------|-------------------|
| Focal Tversky      | 0.386 +- 3.43e-02 | 0.333 +- 7.39e-03 | 0.364 +- 9.64e-03 | 0.322 +- 4.21e-03 |
| Dokładność         | 0.934 +- 9.97e-03 | 0.956 +- 2.28e-03 | 0.947 +- 2.25e-03 | 0.956 +- 1.94e-03 |
| Średnia dokładność | 0.776 +- 2.42e-02 | 0.837 +- 8.81e-03 | 0.807 +- 9.20e-03 | 0.833 +- 7.65e-03 |
| Precyzja           | 0.568 +- 4.68e-02 | 0.691 +- 1.84e-02 | 0.631 +- 1.99e-02 | 0.681 +- 1.60e-02 |
| Czułość            | 0.814 +- 3.28e-02 | 0.812 +- 1.48e-02 | 0.803 +- 2.36e-02 | 0.831 +- 1.04e-02 |
| F1/Dice            | 0.666 +- 3.85e-02 | 0.745 +- 8.41e-03 | 0.705 +- 6.68e-03 | 0.748 +- 6.65e-03 |
| IoU                | 0.504 +- 4.15e-02 | 0.594 +- 1.07e-02 | 0.544 +- 7.92e-03 | 0.598 +- 8.43e-03 |
| ROC AUC            | 0.895 +- 1.35e-02 | 0.969 +- 1.61e-03 | 0.897 +- 1.90e-02 | 0.933 +- 6.79e-03 |

![png](report_images/output_109_0.png)
    
![png](report_images/output_110_0.png)
    
## 6. Generate images
## 6.1. Samsung images

![png](report_images/output_117_0.png)
    
![png](report_images/output_117_1.png)
    
### 6.1.1 Samsung models
**The best**
![png](report_images/output_120_0.png)

**The worst**
![png](report_images/output_122_0.png)

### 6.1.2 GE models
**The best**
![png](report_images/output_125_0.png)
    
**The worst**
![png](report_images/output_127_0.png)
    

### 6.1.3. Mix models
**The best**    
![png](report_images/output_130_0.png)
    
**The worst**
![png](report_images/output_132_0.png)
    

## 6.2. GE images
    
![png](report_images/output_135_0.png)
    
![png](report_images/output_135_1.png)
    

### 6.2.1. GE models
**The best**
![png](report_images/output_138_0.png)

**The worst**
![png](report_images/output_140_0.png)

### 6.2.2. Samsung models
**The best**
![png](report_images/output_143_0.png)
    
**The worst**
![png](report_images/output_145_0.png)
    
### 6.1.3. Mix models
**The best**
![png](report_images/output_148_0.png)

**The worst**    
![png](report_images/output_150_0.png)
