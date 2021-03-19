# [CVPR] Fast and Accurate Model Scaling , 2021
**Piotr Dollar , Mannat Singh , Ross Girshick [FaceBook AI Research]**

## Term 

1. Width : 
2. Depth
3. Resolution
4. FLOPS
5. Parameter
6. Activations


## Abstract

* 본래 일반적으로 Deep Learning 모델에는 속도가 빠르면 정확성이 떨어지고 , 정확성이 올라가면 속도가 느려지는 **Trade-off** 관계가 존재합니다. 

* 그래서 속도와 정확도를 둘다 적당히 잡아주는 방법이 필요했는데 여기에 사용되는 방법이 **Model Scaling**이라고 합니다.

* CNN model의 **"width" , "depth" , "resolution"** 을 조절 해주는 model scaling 방법들이 존재하고 있지만 scaling 방법들의 trade-off 관계가 완벽히 연구되지 않았다고 합니다.

* 현재 대부분의 연구들은 **accuracy**와 **Flops**간의 상호관계에 주로 초점을 두었었습니다.

* 하지만 본 Paper 에서는 **정확도는 유지한채 runtime에서 빠르게 동작할 수 있는** scaling 기법을 다루고 있습니다.

* 미리 얘기하면 본 Paper에서는 **Activation의 수를 최소한으로 증가시키면서** 모델을 scaling 해주는 방법을 제시하는데 ,이는 **width에 초점을 맞추고 Depth와 resolution에는 낮은 비중으로** scaling 하는 방법이 정확성은 비슷하지만 속도를 많이 증가 시켰다고 합니다.


## Introduction


## Related Work


## Complexity of Scaled Models


## Runtime of Scaled Models


## Fast Compound Model Scaling


## Experiments
