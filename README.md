# Generative Distribution Distillation (GenDD)

In this paper, we formulate the knowledge distillation (KD) as a conditional generative problem and propose the Generative Distribution Distillation (GenDD)
framework. A naive GenDD baseline encounters two major challenges: the curse
of high-dimensional optimization and the lack of semantic supervision from labels. To address these issues, we introduce a Split Tokenization strategy, achieving stable and effective unsupervised KD. Additionally, we develop the Distribution Contraction technique to integrate label supervision into the reconstruction objective. Our theoretical proof demonstrates that GenDD with Distribution
Contraction serves as a gradient-level surrogate for multi-task learning, realizing efficient supervised training without explicit classification loss on multi-step
sampling image representations. To evaluate the effectiveness of our method, we
conduct experiments on balanced, imbalanced, and unlabeled data. Experimental
results show that GenDD performs competitively in the unsupervised setting, significantly surpassing KL baseline by **16.29%** on ImageNet validation set. With
label supervision, our ResNet-50 achieves **82.28%** top-1 accuracy on ImageNet
in 600 epochs training, establishing a new state-of-the-art.

## Envrionment
See piplist.txt

## Experimental Results 
Pretrained models will be available soon.   

### Supervised KD on ImageNet

 | Method | Model | Top-1 Acc(%) | Linear prob | link | log | 
 | :---: | :---: | :---: | :---: | :---: | :---: |
 | KD     | ResNet34-ResNet18 | 71.24 | - | - | - |
 | IKL-KD | ResNet34-ResNet18 | 71.91 | - | - | - |
 | GenDD  | ResNet34-ResNet18 | **72.38** | - | [model](https://drive.google.com/file/d/1hVDVDf0e13fisdsVfb6aNwY3371GqbjO/view?usp=sharing) | [log](https://drive.google.com/file/d/1pn8fHOVnvD4cpO6BuQpprU6Ht8soLN2k/view?usp=sharing) |
 | ---    | --- | --- | --- | --- |
 | KD | ResNet50-MVNet | 71.44 | - | - | - |
 | IKL-KD | ResNet50-MVNet | 73.19 | - | - | - | 
 | GenDD | ResNet50-MVNet | **73.78** | - | [model](https://drive.google.com/file/d/1e9kcyBqPuYlyg9aqBwf-SwzTdQpGRNoP/view?usp=sharing) | [log](https://drive.google.com/file/d/1k9_7CH0IK7VOZ-s3gRDNeoa0pR72MO1t/view?usp=sharing) |
 | --- | --- | --- | --- | --- |
 | KD | BEiT-L-ResNet50 (A2 300e) | 80.89  | - | - | - |
 | DKD | BEiT-L-ResNet50 (A2 300e)| 80.77 | - | - | - |
 | GenDD | BEiT-B-ResNet50 (A2 300e) | **82.01**  | - | [model]() | [log](https://drive.google.com/file/d/1lzAowLVxNlzy1WEuPeBWtYiaHLjbkUaZ/view?usp=sharing) |
 | GenDD | BEiT-L-ResNet50 (A2 300e) | **81.83** | - | [model](https://drive.google.com/file/d/1QDJh9SgqyUj0TZFk1dzIgUE4UpxJNWVj/view?usp=sharing) | [log](https://drive.google.com/file/d/1bKuokvl7snVHgSd-lg_qawKTtrYbyXVb/view?usp=sharing) |
 | --- | --- | --- | --- | --- |
 | KD | BEiT-L-ResNet50 (A1 600e) | 81.68  | - | - | - |
 | DKD | BEiT-L-ResNet50 (A1 600e)| 81.83 | - | - | - |
 | GenDD | BEiT-L-ResNet50 (A1 600e) | **82.31** | - | [model](https://drive.google.com/file/d/1xyV16MzXFrpzMLnSYxNq6VmX0iPy5JzW/view?usp=sharing) | [log](https://drive.google.com/file/d/1GBB1haWkHbN-0riMOvckPF5VGAwEWfIo/view?usp=sharing) |
 
### Unsupervised KD on CC3M
We train models on CC3M without labels and evaluate the trained models on ImageNet validation set.               

 | Method | Model | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: |
 | KL | ResNet50-MVNet | 51.60 | - | - |
 | GenDD  | ResNet34-ResNet18 | 66.90 | - | - |
 | GenDD | ResNet50-MVNet | **67.89** | - | - |
 


## Training and Evaluation
Before evaluation, please specify the path of the trained models.              

For CIFAR,   
```
cd GenDD_cifar             
bash sh/fetch_pretrained_teachers.sh            
bash sh/train_res110_res32_gendd.sh              
bash sh/evaluate.sh
```

For supervised KD on ImageNet,    
```
cd GenDD_imagenet              
bash sh/imagenet_train_res34res18_gendd.sh            
bash sh/imagenet_eval_res34res18_gendd.sh                 
```

For unsupervised KD on CC3M,     
```
cd GenDD_imagenet             
bash sh/cc3m_train_res34res18_unsupervised_gendd.sh
bash sh/cc3m_eval_res34res18_unsupervised_gendd.sh
```

## Contact
If you have any questions, feel free to contact us through email (jiequancui@gmail.com) or Github issues. Enjoy!


