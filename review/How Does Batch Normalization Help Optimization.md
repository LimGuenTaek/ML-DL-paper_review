# [NIPS] How Does Batch Normalization help Optimization(2018 , Shibani Santurkar et al)

---

### Before Review

첫번째 Paper review 입니다. 주제는 Batch Normalization으로 준비를 했고 본 paper는 Batch Normalization에 통상적으로 설명되던 주장을 반박하는 논문입니다. 

BatchNorm을 아예 모르신다면 [**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**](http://proceedings.mlr.press/v37/ioffe15.pdf) 논문을 가볍게 읽어보시는 것을 추천합니다.

개인적으로 논문을 공부할 때 태도를 깨우쳐준 논문이라 의미가 있습니다.

BatchNorm이 2015년에 등장을 하고 정말 많은 관심을 받으며 deep learning model에 적용이 되었는데 , 본 paper에서는 2015년에 발표된 Batch Norm의 설명을 뒤엎고 새로운 관점에서 BatchNorm의 이유를 설명하고 있습니다.

기존의 주장이 반박되는 전개과정 속에서 나름 흥미진진하게 읽었고 , 이러한 경험을 통해 논문에 주장되는 설명들이 절대적 진리가 아닐 수도 있다는 사실을 깨닫게 된 논문인 것 같습니다.

아직은 기본기가 많이 부족하기 때문에 다른 논문들을 읽을 때 비판적인 분석을 하는 것은 힘들 수 도 있겠지만 최대한 비판적인 관점을 가지려고 하는 방향으로 성장을 해보겠습니다.

---

### Introduction and motivation

* 지난 십년동안 deep learning 분야에서는 수 많은 발전들이 있었으며 그 중 BatchNorm 또한 분명한 예시가 될만한 성능을 보여주며 자리 잡았습니다.

* 현재 deep learning model에서는 거의 대부분 BatchNorm을 적용하고 있지만 , BatchNorm의 성능이 어디로 부터 나타나는 것인지에 대한 이해는 사실 상 부족했습니다.

* 대부분 알고 있던 이유는 BatchNorm이 각 layer들에 전달되는(Hidden Layer) Input-data 들의 distribution이 달라지는 Internal Covariate shift 라는 현상을 줄여줌으로써 잘 작동한다고 설명하고 있었습니다.

* 하지만 논문의 저자는 이것 또한 확실한 근거가 되지 못한다고 말하고 있습니다.

##### 본 논문의 저자가 밝히는 본 paper의 Contribution은 다음과 같습니다.

  1. 사실 , BatchNorm과 Internal Covariate shift는 관계가 별로 없다

  2. 애초에 BathcNorm이 Internal Covariate shift를 줄여주는 지도 불명확하다.

  3. 우리는 BatchNorm의 좀 더 근본적인 이유를 제시한다

---

### Batch normalization and internal covariate shift

저자는 BathcNorm과 ICS와의 관계에서 다음과 같은 질문을 품게 됩니다.

  - **Is the effectiveness of BatchNorm indeed related to internal covariate shift?**
  - **Is BatchNorm’s stabilization of layer input distributions even effective in reducing ICS?**

다음과 같은 의문을 해결하기 위해 몇가지 실험을 진행합니다.

첫번째 실험은 Batch Norm을 적용한 이후 **random noise**를 첨가해 distribution을 흐트러 놓는 실험입니다. 

이러한 noise가 covariate shift를 야기시키고 , distribution의 skewness를 증가 시킴으로써 결국 **Internal Covariate shift**를 증가시키게 됩니다.

  * case1 : Standard 

  * case2 : BatchNorm + Standard

  * case3 : BatchNorm + Noise

본래 주장으로는 Random noise를 추가해 ICS가 증가했으니 case1 이 case3 보다는 더 좋은 성능을 보여야 합리적인데 예상치 못한 결과가 등장합니다.

<img width="597" alt="스크린샷 2021-03-14 오전 1 26 27" src="https://user-images.githubusercontent.com/70448161/111039130-c8cc4a80-846f-11eb-94e3-437a483caa11.png">

결과를 설명하자면 빨간색으로 표시된 ICS가 증가한 case이지만 BatchNorm이 적용된 모델이 기존 Standard 모델보다 더 좋은 성능을 보이고 있습니다.  이러한 결과는 다음과 같은 의심을 만들어 냅니다.
 
  **“BatchNorm의 성능이 ICS를 줄임으로써 기인하는 것이 아니구나.”**
  
그리고 저자는 다음의 실험 또한 진행합니다. **“BatchNorm을 통해 ICS가 정말로 감소하는 걸까”** 를 알아보기 위해 gradient를 확인합니다.

<img width="742" alt="스크린샷 2021-03-14 오전 1 45 40" src="https://user-images.githubusercontent.com/70448161/111039157-f1ecdb00-846f-11eb-8b4c-0d336e7acb6e.png">

1. 우선 원래대로 Forward 과정에서 k 번째 Layer가 있다고 할 때 K 번째 Layer의 gradient를 계산합니다.

2. 그 다음 BackPropagation 과정에서 K 번째 Layer 이후의 Layer들은 update를 시키지 않고 K 번째 이전의 Layer들 (1 , 2 ,  …  , K-1)들을 update 한후 다시 Foward를 진행해서 K번째 Layer의 gradient를 계산합니다.

이렇게 얻은 두 Gradient를 비교함으로써 ICS의 척도를 확인할 수 있습니다.**(두 Gradient가 비슷하다면  ICS가 감소 됐다고 받아들일 수 있고 , 두 Gradient가 다른 방향성을 가진다면 ICS가 증가했다고 받아들일 수 있습니다.)**

<img width="711" alt="스크린샷 2021-03-14 오전 1 26 37" src="https://user-images.githubusercontent.com/70448161/111039168-06c96e80-8470-11eb-8576-573409116978.png">

(a) VGG network에서는 사실상 비슷하다고 볼 수 있고 , (b) DLN에서는 오히려 더 안좋아진 것을 확인할 수 있습니다. (Cos Angle , 두 Gradient의 사잇각이 클 수록 0에 가까워지고 이는 두 벡터가 비슷하지 않다는 것을 의미합니다.)

저자가 실행한 두 실험은 결국 이러한 사실을 의미합니다. **“BatchNorm의 성능은 ICS에 기인한 것이 아닐 뿐더러 BatchNorm은 ICS를 줄여주지 않는다.”**

---

### Why does BatchNorm work?

BatchNorm이 ICS와 관계가 없다는 사실은 확인 했으나 , BatchNorm은 여전히 좋은 Performance를 보여주고 있습니다. 그렇다면 정말 근본적인 이유는 무엇일까 저자는 고민했고 **Smoothing Effect** 를 주장합니다.

저자가 주장하는 BatchNorm에 대한 설명은 “ Batch Norm이 optimization landscape를 확실하게 smooth 하게 만들어준다. 따라서 gradient들이 predictive 해지며 결국 learning rate를 크게 잡아도 안정적으로 학습을 할 수 있다” 이렇게 말하고 있습니다.

이게 무슨 말이냐면 , Landscape가 smooth 해진다는 말은 Loss 함수의 Lipschitzness(함수의 어느 두점을 잡아도 기울기의 상한이 존재함) 성질이 좋아진다는 의미입니다.

Loss함수가 Lipschitz하다면 결국엔 gradient의 upper-bound가 존재하므로 gradient들이 전반적으로 비슷하게 좀 더 예측 가능하게 존재하므로 학습의 step을 크게 크게 잡아도 무리가 되지 않는다는 뜻입니다. 

**즉 , HyperParameter 설정에 robust 해지면서 train 속도가 빨라지게 됩니다.**

<img width="923" alt="스크린샷 2021-03-14 오전 1 56 27" src="https://user-images.githubusercontent.com/70448161/111039194-206ab600-8470-11eb-9764-6e50dcb79db6.png">

위의 주장을 입증하기 위해 저자는 몇가지 지표를 제시합니다.

여기서 step은 epoch가 아니라 특정 시점에서 gradient 방향으로 얼마나 나아갈 것인지에 대한 지표입니다. step을 많이 밟을 수록(gradient 방향으로 멀리 나아가는 것) 신뢰성이 떨어지므로 변동성이 커지는 것이 자연스러운데 BatchNorm을 적용하면 이런 Loss , gradient의 변동성이 크게 줄어들게 되는 것을 확인할 수 있습니다. 즉 , 크게 step을 밟아도 변동성이 적으므로 우리에게 confidence를 주게 되며 학습이 전반적으로 안정성을 갖게 됩니다.

<img width="748" alt="스크린샷 2021-03-14 오전 1 26 53" src="https://user-images.githubusercontent.com/70448161/111039210-311b2c00-8470-11eb-90b6-4890b4866fd6.png">

다음으로 저자는 BatchNorm 뿐만 아니라 Smoothing 효과를 줄 수 있는 다른 task 들 또한 비슷한 개선을 이룰 수 있다고 보여줍니다.

<img width="715" alt="스크린샷 2021-03-14 오전 2 14 00" src="https://user-images.githubusercontent.com/70448161/111039213-34161c80-8470-11eb-85d4-8869741e38a7.png">

BatchNorm 뿐 아니라 L-normalization 또한 비슷한 Performance를 보여주는 것을 보여주고 있습니다.

---

### Conclusion

본 paper의 contribution은 기존에 설명되었던 BatchNorm과 ICS와의 관계를 실험을 통해 반박을 하였으며 BatchNorm에 대한 새로운 시각으로 원인을 규명하는데에 있습니다.

부록 부분에 Lipschitzness 와 optimzation관련 된 수학적인 증명들이 나와있는데 보다 더 엄밀하게 확인하고 싶으신 분들은 찾아보는 것도 좋을 것 같습니다.






