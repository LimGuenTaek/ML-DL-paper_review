# [CVPR] Fast and Accurate Model Scaling (2021 , Piotr Dollar , Mannat Singh , Ross Girshick )

## Before Review

이번 Review의 주제는 **Model scaling** 입니다. 

**Model scaling**을 아예 모르신다면 **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks(2019 , Mingxing Tan)** 을 읽어보는 것을 추천합니다. 개념을 다지는데 도움이 많이 되었습니다.

본격적인 Review를 시작하기전에 EfficientNet에 있는 자료를 하나 참고하겠습니다.

![scaling](https://user-images.githubusercontent.com/70448161/111809977-dfafe880-8918-11eb-8cbd-9bf7aca0c0aa.PNG)

기존 연구의 큰 흐름을 보여주고 있는 그림입니다. 

Scaling은 Filter의 채널수(**width**)를 늘리거나 , Layer수(**Depth**)를 늘리거나 , Input-size(**Resolution**)를 늘려주는 방법들로 구성이 됩니다.

알고 가면 더욱 이해가 잘 될 것 같아 언급하고 Review를 시작하겠습니다.

## Introduction and Motivation

* 본래 일반적으로 Deep Learning 모델에는 속도가 빠르면 정확성이 떨어지고 , 정확성이 올라가면 속도가 느려지는 **Trade-off** 관계가 존재합니다. 

* 그래서 속도와 정확도를 둘 다 적당히 잡아주는 방법이 필요했는데 여기에 사용되는 방법이 **Model Scaling**이라고 합니다.

* CNN model의 **"width" , "depth" , "resolution"** 을 조절 해주는 model scaling 방법들이 존재하고 있지만 scaling 방법들의 trade-off 관계가 완벽히 연구되지 않았다고 합니다.

* 대부분의 기존 연구들은 **accuracy**를 향상시키는 데에 초점을 두었었습니다.

* 그래서 본 연구의 Motivation은 다음과 같습니다. **"Can we design scaling strategies that optimize both accuracy and model runtime(speed)?"**
 
* 본 연구는 Fast scaling이라는 방법을 제시했고 , 이는 **동일한 Accuracy지만 더 빠른 Runtime을 가질 수 있게하는 Scaling 방법**입니다.

* 핵심은 **Activation을 최소한으로 증가시키면서** 모델을 scaling 해주는 방법을 제시하는데 ,이는 **width에 초점을 맞추고 Depth와 resolution에는 낮은 비중으로** scaling 하는 방법입니다.

<img width="862" alt="스크린샷 2021-03-19 오후 4 04 49" src="https://user-images.githubusercontent.com/70448161/111744190-2c220680-88ce-11eb-8441-0f1bf9a57f0c.png">

* 그래프를 보면 정확도 측면에서는 Compound scaling과 비슷하거나 동일한 성능을 보여주지만 , **epoch time**과 **scale증가에 따른 activation의 점근적 증가율**이 낮은 것을 확인할 수 있습니다.

* scale을 키워줌으로써 activation이 천천히 증가하는 것은 Run-time 측면에서 중요한 역할을 하는데 이유는 뒷부분에서 다루도록 하겠습니다.

* 본 Paper의 아이디어를 살펴보기전에 우선 Model의 complexity를 설명하는 지표에 대해 간단히 알아보겠습니다. (flops , parameters , activations)

## Complexity of Scaled Models


## Runtime of Scaled Models

* 다시 한번 본 Paper의 Motivation을 상기시켜 보면 , **fast 하면서 accurate한** scaling 방법을 고안하는 것이었습니다.

* 위에서 다양한 scaling 기법에 대한 flops , parameter , activation들의 변화를 살펴봤습니다.

* 여기서는 어떠한 요소가 model run-time에 **가장 밀접한 관계가 있는지** 알아보겠습니다.

* 밑에 그림을 보면 flops , parameters , activations 에 대한 run-time을 각각의 scaling 기법을 적용해 표현한 그래프입니다.

<img width="679" alt="스크린샷 2021-03-19 오후 6 13 43" src="https://user-images.githubusercontent.com/70448161/111757634-df92f700-88de-11eb-8867-19a0181f6727.png">

  그래프를 간단히 설명하면 EN-OO 라고 적혀있는 것은 Efficient-Net에 4가지 scaling 기법을 적용한 것입니다.(single-width , depth-width , Compound , fast-Compound)
  
  fit 이라고 적힌 검정색 line은 **Pearson Correlation** 를 계산한 것이며 **flops , parameters , activations** 들과 **Runtime** 간에 **관련성**을 나타내줍니다.
  
  하나씩 확인을 해보자면 먼저 **Flops**의 Pearson Coefficient는 0.81 입니다. 서로 비슷한 추이를 가지고 있는 것 같지만 Flops가 Runtime을 완전히 예측할 수는 없습니다. Scaling 기법별로 증가하는 추이가 조금 다르기 때문입니다.
  
  다음은 **Parameter**입니다. Parameter는 Flops 보다 더 낮은 Pearson Coefficient를 가지고 있으며 확실히 scaling 기법별로 증가하는 추이가 완전히 다른것을 확인 할 수 있습니다.
  
  마지막으로 **Activation**입니다. Activation은 scaling 기법과는 관계없이 일관된 추이를 보여주고 있습니다. 따라서 Activation으로 Runtime을 예측하는 것은 좋은 접근입니다.
  
  확인해보면 Activation이 Runtime과 **가장 높은 관련성**을 보여주고 있다는 것을 확인할 수 있습니다.
  
  **즉 , Activation의 증가가 Runtime에 더욱 치명적이다 라는 의미를 내포하고 있습니다.**

## Fast Compound Model Scaling

  * 앞선 실험에서 Model의 Run-time과 activation이 큰 연관 **(scaling 기법 상관없이 activation이 증가하면 Runtime이 증가한다)** 을 가지는 것을 확인했습니다. 
  
  * 그래서 저자는 scaling 방식을 **가능한 activation을 최소화시키는 방향**으로 고안했습니다. 
  
  * 아이디어는 Single-scaling 결과를 관찰한 것에서 기인합니다. Width가 그나마 activation 증가에 덜 영향을 주기 때문에 Width scaling에 비중을 높이고 , Depth와 Resolution에 비중을 줄이는 방향으로 설계 합니다.
---
변수 alpha를 도입하여 이러한 아이디어를 구현할 수 있는데 먼저 어떤 상수들을 정의 해줍니다. 

<img width="347" alt="스크린샷 2021-03-19 오후 6 50 09" src="https://user-images.githubusercontent.com/70448161/111762179-0f90c900-88e4-11eb-84fa-c232cb340a0c.png">
  
 그리고 scaling factor들을 위에서 정의한 상수를 이용해서 다음과 같이 정의해줍니다.

<img width="416" alt="스크린샷 2021-03-19 오후 6 50 17" src="https://user-images.githubusercontent.com/70448161/111762185-10c1f600-88e4-11eb-9484-799cfecb57ba.png">

 이제 위에서 정의한 scaling factor들을 가지고 flops , parameters , activations 들을 계산해줍니다.

<img width="438" alt="스크린샷 2021-03-19 오후 8 11 21" src="https://user-images.githubusercontent.com/70448161/111771657-4b7d5b80-88ef-11eb-8d9f-36b875f08ad0.png">
   
이러한 alpha들을 조절해줌으로써 다양한 결과들이 나온다고 합니다

   1. alpha = 0 : resulting in depth and resolution scaling (dr)
    
   2. alpha = 1/3 : corresponds to uniform compound scaling (dwr)

이때 , 흥미로운 부분이 1/3 < alpha < 1 부분인데 , alpha가 1에 가까우면(width에 비중을 높게 준다는 의미) **fast scaling**이 된다고 합니다 

아래 Table을 확인하면 activation이 비교적 적게 증가하는 것을 확인할 수 있습니다.

<img width="396" alt="스크린샷 2021-03-19 오후 6 50 41" src="https://user-images.githubusercontent.com/70448161/111772310-102f5c80-88f0-11eb-8e32-462832a9f3e1.png">


## Experiments
