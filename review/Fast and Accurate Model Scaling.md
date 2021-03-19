# [CVPR] Fast and Accurate Model Scaling , 2021
**Piotr Dollar , Mannat Singh , Ross Girshick [FaceBook AI Research]**


## Abstract

* 본래 일반적으로 Deep Learning 모델에는 속도가 빠르면 정확성이 떨어지고 , 정확성이 올라가면 속도가 느려지는 **Trade-off** 관계가 존재합니다. 

* 그래서 속도와 정확도를 둘다 적당히 잡아주는 방법이 필요했는데 여기에 사용되는 방법이 **Model Scaling**이라고 합니다.

* CNN model의 **"width(filter의 채널갯수)" , "depth(layer들의 갯수)" , "resolution(input size)"** 을 조절 해주는 model scaling 방법들이 존재하고 있지만 scaling 방법들의 trade-off 관계가 완벽히 연구되지 않았다고 합니다.

* 현재 대부분의 연구들은 **accuracy**와 **Flops**간의 상호관계에 주로 초점을 두었었습니다.

* 하지만 본 Paper 에서는 **정확도는 유지한채 runtime에서 빠르게 동작할 수 있는** scaling 기법을 다루고 있습니다.

* 미리 얘기하면 본 Paper에서는 **Activation의 수를 최소한으로 증가시키면서** 모델을 scaling 해주는 방법을 제시하는데 ,이는 **width에 초점을 맞추고 Depth와 resolution에는 낮은 비중으로** scaling 하는 방법이 정확성은 비슷하지만 속도를 많이 증가 시켰다고 합니다.


## Introduction

  - 기존의 Scaling 연구들은 대부분 Accuracy를 높여주는 것이 대부분이었습니다.
  
  - 그래서 본 Paper의 저자는 다음과 같은 의문이 들었다고 합니다 **"Can we design scaling strategies that optimize both accuracy and model runtime(speed)"**

  - 본 연구는 Fast scaling이라는 방법을 제시했고 따라서 본 논문의 Contribution은 **동일한 Accuracy지만 더 빠른 Runtime을 가질 수 있게하는 Scaling 방법을 제시한 것에 있습니다.**

<img width="862" alt="스크린샷 2021-03-19 오후 4 04 49" src="https://user-images.githubusercontent.com/70448161/111744190-2c220680-88ce-11eb-8441-0f1bf9a57f0c.png">

그래프를 보면 정확도 측면에서는 Compound scaling과 비슷하거나 동일한 성능을 보여주지만 , epoch time과 activation의 점근적 증가율이 많이 낮은 것을 확인할 수 있습니다.

어떻게 이런 성능에 도달했는지는 Fast Compound Model Scaling 파트에서 다루도록 하고 우선 다양한 scaling 기법의 complexity를 분석하는 방법에 대해 알아보겠습니다. (flops , parameters , activations)

## Complexity of Scaled Models

이 파트에서는 다양한 network scaling 기법의 complexity를 분석하는 방법에 대해 알아보겠습니다.

#### Complexity Metrics


## Runtime of Scaled Models

* 다시 한번 본 Paper의 Motivation을 상기시켜 보면 , **fast 하면서 accurate한** scaling 방법을 고안하는 것이었습니다.

* 위에서 다양한 scaling 기법에 대한 flops , parameter , activation들의 변화를 살펴봤습니다.

* 여기서는 어떠한 metric이 model run-time에 **가장 밀접한 관계가 있는지** 알아보는 것입니다.

* 밑에 그림을 보면 flops , parameters , activations 에 대한 run-time을 각각의 scaling 기법을 적용해 표현한 그래프입니다.

<img width="679" alt="스크린샷 2021-03-19 오후 6 13 43" src="https://user-images.githubusercontent.com/70448161/111757634-df92f700-88de-11eb-8867-19a0181f6727.png">

  그래프를 간단히 설명하면 EN-OO 라고 적혀있는 것은 Efficient-Net에 4가지 scaling 기법을 적용한 것입니다.(single-width , depth-width , Compound , fast-Compound)
  
  fit 이라고 적힌 검정색 line은 **Pearson Correlation** 를 계산한 것이며 **flops , parameters , activations** 들과 **Runtime** 간에 **관련성**을 나타내줍니다.
  
  확인해보면 Activation이 Runtime과 **가장 높은 관련성**을 보여주고 있다는 것을 확인할 수 있습니다.

## Fast Compound Model Scaling

  * 앞선 실험에서 Model의 Run-time과 activation이 큰 연관 **(activation이 증가하면 Runtime이 증가한다)** 을 가지는 것을 확인했습니다. 
  
  * 그래서 저자는 scaling 방식을 **가능한 activation을 최소화시키는 방향**으로 고안했습니다. 
  
  * 아이디어는 Single-scaling 결과를 관찰한 것에서 기인합니다. Width가 그나마 activation 증가에 덜 영향을 주기 때문에 Width scaling에 비중을 높이고 , Depth와 Resolution에 비중을 줄이는 방향으로 설계 합니다.

<img width="347" alt="스크린샷 2021-03-19 오후 6 50 09" src="https://user-images.githubusercontent.com/70448161/111762179-0f90c900-88e4-11eb-84fa-c232cb340a0c.png">
  
    변수 alpha를 도입하여 이러한 아이디어를 구현할 수 있는데 먼저 간단한 상수들을 정의 해줍니다. 
  
<img width="416" alt="스크린샷 2021-03-19 오후 6 50 17" src="https://user-images.githubusercontent.com/70448161/111762185-10c1f600-88e4-11eb-9484-799cfecb57ba.png">

    그리고 scaling factor들을 다음과 같이 정의해줍니다.
  
<img width="438" alt="스크린샷 2021-03-19 오후 8 11 21" src="https://user-images.githubusercontent.com/70448161/111771657-4b7d5b80-88ef-11eb-8d9f-36b875f08ad0.png">
 
    위에서 정의한 scaling factor들을 가지고 flops , parameters , activations 들을 정의해줍니다.
  
이러한 alpha들을 조절해줌으로써 다양한 결과들이 나온다고 합니다

* alpha = 0 : resulting in depth and resolution scaling (dr)
* alpha = 1/3 : corresponds to uniform compound scaling (dwr)

이때 , 흥미로운 부분이 1/3 < alpha < 1 부분인데 , alpha가 1에 가까우면 **fast scaling**이 된다고 합니다 

본 Paper에서는 **alpha=0.8** 을 default로 설정했습니다. (dWr)

<img width="396" alt="스크린샷 2021-03-19 오후 6 50 41" src="https://user-images.githubusercontent.com/70448161/111772310-102f5c80-88f0-11eb-8e32-462832a9f3e1.png">

**activation이 비교적 적게 증가하는 것을 확인할 수 있습니다.**

## Experiments
