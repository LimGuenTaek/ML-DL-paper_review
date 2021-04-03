# [NIPS] Generative Adversarial Network (2014 , Ian J. Goodfellow et al)

## Before Review

오늘 리뷰할 Paper는 그 유명한 GAN 입니다.

주워듣기만 하다가 논문이랑 관련 자료들을 참고해서 GAN에 대해 나름대로 공부를하고 정리를 해보려고 합니다.

읽으면서 정말 간단하면서도 명료한 아이디어에 감탄하게 됐고 , 수식 또한 복잡하지 않아 이해하는데 큰 어려움이 없었습니다.

여담으로 GAN의 탄생 배경에 대해 조금 알아봤는데 당시에도 인간의 뇌를 흉내낸 신경망을 이용해 그럴듯한 데이터를 스스로 만드는 생성적 모델을 사용하고 있었지만 성능은 그리 좋지 않았다고 합니다. 

복잡한 통계적 분석 기법을 각 요소들에 적용하는 방법들이 고려되고 있었는데 이는 수많은 계산이 필요한 일이였고 , 저자인 이안 굿펠로우는 이러한 방법이 좋지 않은 접근이라 생각하고 있었습니다.

맥주를 마시며 이 문제를 고민하던 이안 굿 펠로우는 마침내 "두개의 신경망을 서로 경쟁하면 어떨까" 라는 아이디어가 떠올랐고 집에 돌아온 후 몇시간동안 코딩을 한 결과 GAN이라는 멋진 결과가 나왔다고 합니다.(천재인듯 합니다..)

이렇게 2014년에 GAN이 세상에 등장한 이후 정말 많은 후속연구가 파생되었는데 오늘 그 시초가 되었던 GAN에 대해 review 해보겠습니다.

## Introduction

사실 , GAN이 나오기 전에도 Generative Model들이 존재했지만 , GAN이 지금까지도 각광받는 이유는 **적대적** 생성 모델이기 때문인 것 같습니다.

**적대적**이란 말을 설명하기전에 우선 Generative Adversarial Network에 존재하는 **두가지 network**를 간단히 소개하겠습니다.

* **Generative Model** 
   * 우리가 Input 으로 넣어준 Data(image, voice, text)의 distribution을 알아내려고 노력하는 network 입니다. 
   * 바로 이 Generator가 그럴듯한 데이터들을 생성해주는 생성 모델입니다.
   * 이 Generator가 Data의 distribution을 잘 학습 했다면 학습한 distribution과 noise vector를 섞어줘서 그럴듯한 생성 data instance를 만들어 줄 수 있습니다.

* **Discriminative Model** 
   * 입력으로 들어온 sample data가 진짜 input data로 부터 온 것인지 Generator로 부터 온 것인지를 구별하여 각각의 경우에 대한 확률을 estimate 합니다.
   * 입력으로 들어온 sample data가 Discriminator가 판단하기에 real data에 가까운 것 같다면 확률값은 1 , fake data에 가까운 것 같다면 확률값은 0 이런식으로 반환합니다.
   * 즉, 만들어진 가짜 데이터인지 진짜 입력 데이터로부터 온 것인지 구별하는 일을 하며 , 결과적으론 더 정교한 Generator를 만들기 위해 사용됩니다.

이를 가장 쉽게 설명해주는 예시가 위조지폐범과 경찰인데 간단하게 말하면 위조지폐범(Generator)은 경찰의 눈을 피해 진짜 같은 위조지폐를 만들기 위해 노력하고 경찰(Discriminator)는 더욱 더 정교한 기법으로 지폐들을 검출하는 방향으로 노력합니다.

이것이 GAN의 핵심 아이디어 입니다. 

Generator는 더욱 더 그럴듯한 Data를 만들기 위해 노력하고 Discriminator는 Data를 더욱 더 정확하게 구별하기 위해 노력합니다. 

결국 적대적 생성 모델이라는 것은 두개의 Network가 경쟁을 통해 정확해지고 이때 우리가 원하는 성능 좋은 Generator를 얻을 수 있습니다.

Generator 와 Discriminator의 Network를 본 Paper에서는 MultiLayer Perceptron으로 구현했고 후에 CNN이나 다른 network로 구현하는 많은 후속 연구가 나와있습니다.
 
## Adversarial Network

이제 Adversarial Network가 어떻게 구성이 되는지에 대해 알아보겠습니다.

## Theoretical Results

## Advantages and disadvantages
