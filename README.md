Applied Statistics and Data Science, Yerevan State University
Computer Vision Problem Solving Session: YOLOv1

========

This repository implements Yolo, specifically [Yolov1](https://arxiv.org/pdf/1506.02640) with training, inference in PyTorch.
For training [VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html) dataset is used.

For training you can use the command ```python -m train```. 
Additionally, you can provide a config file ```python -m train --config_file_path <config_file_path>```. By default, ```configs/config_resnet18.json``` is used.

You can create your custom configs with different sets of hyperparameters and do experiments. 

To track experiments and monitor metrics, run ```mlflow ui --port <port>``` 

========

Big thanks to ExplainingAI for [codes](https://github.com/explainingai-code/Yolov1-PyTorch) and [video](https://youtu.be/TPD9AfY7AHo)




## Citations
```
@article{DBLP:journals/corr/RedmonDGF15,
  author       = {Joseph Redmon and
                  Santosh Kumar Divvala and
                  Ross B. Girshick and
                  Ali Farhadi},
  title        = {You Only Look Once: Unified, Real-Time Object Detection},
  journal      = {CoRR},
  volume       = {abs/1506.02640},
  year         = {2015},
  url          = {http://arxiv.org/abs/1506.02640},
  eprinttype    = {arXiv},
  eprint       = {1506.02640},
  timestamp    = {Mon, 13 Aug 2018 16:48:08 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/RedmonDGF15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
