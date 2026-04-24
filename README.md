# SimpleMapTRv2
An unofficial implementation of MapTRv2. I hate using mmdet3d and its related libraries even thgouh I find them very useful. That is why I re-implemented MapTRv2. Most of this project based on the original MapTRv2 code, but this version is free from mmdet3d. I've trained this model on nuScenes and successfully reproduced the result (camera only, 24 epochs, resnet50). The following table is my outcome. 

| Divider | Ped Cross | Boundary | Mean |
|---------|-----------|----------|------|
| 61.5    | 60.3      | 63.2     | 61.7 |
