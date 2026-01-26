---
title: "Attention Is All You Need"
categories: [paper]
tags: [paper, AI]
excerpt: Attention Is All You Need 논문 리뷰
use_math: true
show_each_post: false
---

# Abstract
NLP 분야에서 활발히 연구되던 recurrence, convolution 방식을 버리고 오직 attention 기법만 사용하는 transformer를 제안한다. transformer는 이전 기법들보다 성능이 뛰어날 뿐만 아니라 계산을 병렬화할 수 있고, 더 적은 비용과 시간으로 학습시킬 수 있다는 장점이 있다.

# Introduction
language modeling이나 machine translation 분야에서 RNN, LSTM 등이 SOTA 모델이었다. recurrent 모델은 sequential한 특성으로 인해 병렬화가 어려웠다. 계산 기법이 발전하며 계산 효율이 좋아졌지만, recurrent 모델의 근본적인 한계는 여전하다.
 
attention 기법은 input, ouput 문장 안에서 단어 사이의 거리와 상관없이 단어 간의 의존성을 잘 파악할 수 있으며, sequence modeling, transduction model 등에서 필수적인 요소가 되었다.

이 논문에서 Transformer가 attention 기법만 사용하여 recurrence은 피하면서 단어 간의 의존성을 잘 파악하는 모델임을 증명한다. transformer는 효율적인 병렬화가 가능하며 8개의 P100 GPU로 12시간 학습하면 최고의 성능을 낼 수 있는 성능 좋은 구조다.        

# Background
sequential한 모델은 병렬화가 어렵고, 거리가 먼 단어일수록 의존성을 계산하기 위해 많은 연산이 필요하기 때문에 먼 단어 간의 의존성을 학습하기 힘들다. 하지만 transformer는 상수 계산 복잡도로 이를 해결한다. 다만, effective resolution이 줄어드는 단점이 있지만, 이는 Multi-Head Attention으로 보완한다.

Self-attention은 단일 문장 안에서 단어들 간의 연관성을 계산하는 기법이다. transformer는 RNN이나 convolution 없이 self-attention만 사용한다.


# Model Architecture
![](\assets\images\2026-01-24-attention_is_all_you_need\image.png){: width="70%" .align-center}

## Encoder and Decoder Stacks
### Encoder
![](\assets\images\2026-01-24-attention_is_all_you_need\image2.png){: width="70%" .align-center}
![](\assets\images\2026-01-24-attention_is_all_you_need\image3.png){: width="70%" .align-center}
6개의 동일한 레이어가 쌓여있다. 각 레이어는 2개의 서브 레이어로 구성된다. 
![](\assets\images\2026-01-24-attention_is_all_you_need\image4.png){: width="70%" .align-center}

첫 번째는 multi-head self-attention mechanism, 두 번째는 position-wise fully connected feed-forward network다. 그리고 각 레이어를 통과한 뒤에 residual connection와 layer normalization을 순차적으로 통과한다.

### Decoder
![](\assets\images\2026-01-24-attention_is_all_you_need\image5.png){: width="70%" .align-center}

encoder와 마찬가지로 6개의 동일한 레이어로 구성된다. decoder의 각 레이어는 encoder의 서브 레이어와 매우 비슷하지만, multi-head attention을 수행하는 3번째 서브 레이어가 있다.

## Attention
![](\assets\images\2026-01-24-attention_is_all_you_need\image6.png){: width="70%" .align-center}
attention function은 query 벡터와 key-value 쌍 벡터를 조합한다. query와 key가 결합하여 weight가 만들어지고, 이 weight와 value가 결합하여 output이 나온다.

### Scaled Dot-Product Attention
input은 $ d_k $ 차원 벡터인 queries, keys와 $ d_v $ 차원 벡터인 value로 구성된다. 하나의 query와 문장의 모든 단어에 해당하는 key를 dot product하고 $ \sqrt{d_k} $로 나눈다. 그리고 value에 곱한 weight를 구하기 위해 softmax를 적용한다.

$ Attention(Q, K, V ) = softmax( \frac{QK^T}{\sqrt{d_k}})V $

### Multi-Head Attention
scaled dot-product attention을 개별적으로 h번 반복하여 선형적으로 쌓으면 더 성능이 좋다. 이것이 multi-head attention이다. 문장 내의 서로 다른 위치에서 오는 다른 정보를 각각의 subspace를 통해 동시에 파악할 수 있다. 반면 single attention head는 평균을 낸다.

$ MultiHead(Q, K, V) = Concat(head_1, \; ..., \; head_h)W^O \\\ where \; head_i = Attention(QW_{i}^{Q}, \; KW_{i}^{K}, \; VW_{i}^{V}) $

이 연구는 h = 8로 8개의 병렬적인 attention layer로 구성된다.

### Applications of Attention in our Model
transformer에서 multi-head attention은 3가지 방식으로 쓰인다.

- encoder-decoder attention 레이어에서 queries는 이전 decoder 레이어에서 오고, keys와 values는 encoder의 output에서 온다.
- encoder는 self-attention 레이어로 구성되는데, self-attention 레이어의 keys, values, queries는 모두 이전 encoder의 output로부터 생성된다.
- decoder의 self-attention 레이어는 그 위치와 이전 위치의 decoder에 attend 할 수 있도록 한다. 이러한 masking은 scaled dot-product attention 내에서 구현된다. softmax의 input을 $ - \infty $ 로 설정하여 masking 한다.

## Position-wise Feed-Forward Networks
encoder와 decoder 레이어에 있는 fully connected feed-forward network는 2개의 선형 결합이 있고 그 사이에 ReLU 활성 함수가 있다.

$ FFN(x) = max(0, xW_1 + b_1)W_2 + b_2 $

## Embeddings and Softmax
input token과 output token을 $ d_model $ dimension으로 변환하기 위해서 learned embedding을 사용한다. 그리고 decoder의 output을 predicted next-token probabilites로 변환하기 위해서 learned linear transformation와 softmax function을 사용한다.

## Positional Encoding
input embedding에 positional embedding을 더한다. 

$$
PE_{(pos, 2i)} \; = \; sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
\\
PE_{(pos, 2i+1)} \; = \; cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$

$ pos $: 문장 내에서의 위치(0번째 토큰, 1번째 토큰, ...)

$ i $: 임베딩 벡터($ d_{model} = 512 $) 내에서의 인덱스(0, 1, ..., 255)

각 차원(i)마다 서로 다른 주기를 가진 sin, cos 함수를 할당한다. i가 작은 앞쪽 차원은 주기가 짧아 빠르게 값이 바뀌고, i가 큰 뒤쪽 차원은 주기가 길어 값이 천천히 바뀐다.

단순히 위치를 구분하는 것을 넘어, 문장 내에서의 단어의 상대적인 위치를 잘 학습하기 위해서 삼각함수를 사용했다. 이는 삼각함수의 덧셈공식을 생각해보면 되는데 pos + k 만큼 떨어진 위치의 인코딩 벡터는 pos 위치의 벡터에 특정 회전 행렬을 곱한 것, 즉 선형 변환으로 표현될 수 있기 때문이다. 비선형변환보다 선형변환을 잘 학습하는 인공지능 모델의 특성상 삼각함수를 positional encoding으로 사용하는 것을 적절하다.

# Why Self-Attention
self-attention을 써야 하는 이유는 아래와 같다.

- computational complexity per layer
- amount of computation that can be parallelized
- path length between long-range dependencies in the network

path length는 network 내에서 두 단어의 dependency를 계산하기 위해 거쳐야 하는 단계를 의미한다. path length가 크다면 정보가 희석되거나, 역전파할 때 기울기가 소실되는 문제가 생긴다. 이를 long-term dependency 문제라고 부른다.  RNN은 문장의 맨 첫 단어와 맨 끝 단어의 dependency를 학습하기 위해 $ O(n) $번의 단계를 거쳐야 한다. CNN은 $ O(log_kn) $번의 계산이 필요하다. RNN 보다는 적지만 여전히 많은 층을 쌓아야 한다. 반면 transformer는 attention 덕분에 모든 단어가 모든 단어와 직접적으로 연결되어 있어 O(1)번 만에 dependency를 계산할 수 있다. 따라서 긴 문장에서의 학습이 뛰어나다.

# Training
## Training Data and Batching
- English-German: standard WMT 2014 English-German dataset을 사용한다. 4.5백만 개의 문장 쌍으로 구성된다. byte-pair encoding을 사용하며, 두 언어 간에 37,000 toekn으로 구성된 shared source-target vocabulary를 사용한다. 이를 통해 메모리를 절약할 수 있고, 두 언어 사이의 관계를 더 쉽게 파악할 수 있다.
- English-French: 훨씬 큰 WMT 2014 English-French dataset을 사용한다. 36만 개의 문장으로 구성되고, word-piece 방식으로 32,000 크기의 vocabulary를 만든다.

byte-pair encoding은 vocabulary에 없는 단어를 최소화할 수 있는 encoding 방식이다. 단순히 단어 단위로 encoding 하는 것이 아니라 단어를 더 작은 의미 단위로 쪼갠다. 데이터에서 자주 등장하는 인접한 알파벳 쌍을 찾은 뒤에 병합하는 과정을 거친다. 예를 들어, unfriendly는 un + friend + ly로 분해된다. 이 방식을 사용하면 처음 보는 단어도 아는 조각들의 조합으로 처리할 수 있어서 \<UNK\> 가 거의 발생하지 않는다는 장점이 있다.

word-piece 방식은 BPE와 유사하지만, 무엇을 기준으로 병합하는지를 결정하는 score function이 다르다.

## Hardware and Schedule
8개의 NVIDIA P100 GPU로 구성된 1대의 컴퓨터에서 모델을 학습했다. base model은 100,000 steps를 12시간동안 학습했고, big model은 300,000 steps를 3.5일동안 학습했다.

## Optimizer
Adam optimizer를 사용하고, $ \beta_1 \; = \; 0.9, \; \beta_2 \; = \; 0.98, \; \epsilon \; = \; 10^{-9} $으로 설정했다. learning rate는 처음의 $ warmup_steps(\,=\,4000) $ 동안에는 선형적으로 증가하고, 이후에는 step의 inverse square root에 따라 감소한다. 

$$
learning \; rate \; = \; d_{model}^{-0.5} \, \cdot \, min(step\_num^{-0.5}, \, step\_num \, \cdot \, warmup\_steps^{-1.5})
$$

## Regularization
### Residual Dropout
각 sub-layer의 output이 다음 sub-layer의 input에 들어가거나 normalized 되기 전에 dropout을 적용한다. embedding과 positional encoding의 합에도 dropout을 적용한다.

### Label Smoothing
$ \epsilon_{ls} \; = \; 0.1 $로 label smoothing을 적용했다. 기본적으로 딥러닝 분류 모델은 one-hot encoding 된 정답 label을 가진다. 이것을 hard target이라고 하며, 정답 label이 [0, 0, 1, 0] 과 같이 구성되며, 모델이 정답 클래스에 대해 100% 확신을 갖도록 한다. 하지만 label smoothing을 적용하면 정답 label을 모호하게 만든다. 예를 들어 정답 확률(1.0)에서 0.1만큼을 떼어내서, 나머지 오답들에 골고루 분배한다. 이를 soft target이라고 하며, 정답 label은 [0.025, 0.025, 0.925, 0.025] 처럼 구성된다.

label smoothing을 하면 perplexity(PPL)이 나빠진다. PPL은 다음에 나올 단어를 얼마나 헷갈려 하는가를 나타내는 지표로 최솟값인 1일 때가 가장 완벽히 예측하는 것이고 높아질수록 많이 헷갈린다는 의미다. 수식적으로는 $ e^{cross-entropy loss} $ 형태다. one-hot encoding 일 때는 모델이 정답을 100% 맞히면 loss가 0이 되고 PPL은 1이 된다. 하지만 label smoothing을 쓰면 정답 자체가 불확실하기 때문에 모델의 불확실성을 나타내는 PPL이 높아진다. 

PPL 수치는 나빠졌는데, 번역 성능은 좋아진 이유는 과적합을 방지하기 때문이다. one-hot encoding에서는 softmax의 특성상 정답 클래스의 값을 $ \infty $로 키워야 한다. 이렇게 되면 모델이 하나의 정답에만 집착하게 되어 융통성이 없어진다. 하지만 label smoothing은 정답이 아닌 단어들에도 가능성을 열어둘 수 있다. 이렇게 하면 의미가 비슷한 단어들(ex. happy와 glad) 사이의 관계를 더 유연하게 학습할 수 있다. 결과적으로 일반화 성능이 좋아져서 정확한 변역을 할 수 있다.

# Results
## Machine Translation
WMT 2014 English-to-German traslation task에서 big transformer 모델은 기존 최고 모델을 2.0 BLEU 넘게 뛰어넘으며, SOTA 모델이 되었다. BLEU 점수는 28.4다. base model은 모든 기존 모델과 ensemble을 뛰어넘었고, 경쟁력 있는 모델에 비해 낮은 training cost를 지불했다.

WMT 2014 English-to-French translation task에서 big model은 모든 기존 single 모델을 뛰어넘어 BLEU score 41.0을 달성했고, training cost는 이전 SOTA 모델의 $ 1/4 $ 이하다. 

base model은 5개의 checkpoint를 평균낸 single 모델을 사용했으며, big model은 20개의 checkpoint를 평균냈다.

## Model Variations
아래는 모델의 파라미터를 조정하며 성능을 측정한 표다.
![](\assets\images\2026-01-24-attention_is_all_you_need\image9.png){: width="70%" .align-center}

- single-head attention인 경우 최상의 세팅에 비해 0.9 BLEU 낮았고, head가 너무 많을 때도 마찬가지로 점수가 낮았다.
- attention key size $ d_k $를 줄이는 것도 모델 성능에 악영향을 미쳤다.
- 큰 모델의 성능이 더 좋다.
- dropout이 overfitting을 피하는 데 도움이 된다.
- sinusoidal positional encoding(sin, cos) 대신 learned positional encoding을 썼을 때 거의 비슷한 결과를 얻었다. 성능 차이가 별로 없기 때문에 이 논문에서 learned positional encoding 대신 sinusoidal positional encoding을 쓴다.

## English Constituency Parsing
영어 문장의 구조를 분석하는 task다. transformer가 다른 task로 일반화될 수 있는지 평가하기 위해 시행한 task로 이 task가 어려운 이유는 다음과 같다.

- output에 구조적 제약이 있다.
- input에 비해 output이 훨씬 더 길다.

비록 task-specific tuning을 하지 않았지만, Recurrent Neural Network Grammar 모델을 제외한 모든 모델보다 성능이 좋았다.

또한 RNN Seq2Seq 모델은 데이터셋이 작을 때 과적합이 일어나거나 학습이 잘 안 돼서 성능이 안 좋지만, transformer는 40K 문장으로 이루어진 작은 WSJ 데이터셋에서도 통계적 구문 분석 모델의 SOTA였던 BerkeleyParser 보다 성능이 좋았다.

# Conclusion
이 논문에서 sequence transduction task에서 encoder-decoder 구조를 지닌 모델 중 자주 쓰이던 recurrent layers를 대체할 수 있는 transformer를 제안했다. transformer는 attention만 사용하며 multi-headed self-attention 구조를 가진다.

translation task에서 transformer는 recurrent 또는 convolutional layer 보다 훨씬 더 빠르다. WMT 2014 English-to-German과 WMT 2014 English-to-French translation task에서 SOTA를 달성했다. 첫 번째 task에서는 기존의 모든 ensemble 모델의 성능을 뛰어넘었다.

앞으로의 도전 과제는 input, output을 text 뿐만 아니라 크기가 큰 image, audio, video로 확장시키는 것이다.또한 sequential 생성을 줄이는 것 역시 목표다.


# 기타
- 이 논문의 제목은 2017년 당시 NLP 분야를 지배하고 있던 두 가지 구조인 recurrence와 convolution이 더 이상 필요 없다고 선언하는 의미다. 두 구조의 한계였던 sequential과 locality를 해결할 수 있었다.

- scaled dot-product attention VS self-attention
![](\assets\images\2026-01-24-attention_is_all_you_need\image7.png){: width="70%" .align-center}

- ensemble 기법
앙상블 기법은 여러 개의 모델을 따로 학습시킨 뒤, 그들의 예측 겨로가를 합쳐서 더 좋은 성능을 내는 기법이다. 
![](\assets\images\2026-01-24-attention_is_all_you_need\image8.png){: width="70%" .align-center}

이 논문에서는 여러 개의 모델을 학습시키는 대신에 checkpoint에서의 

# 참고 자료
- https://cpm0722.github.io/pytorch-implementation/transformer
- https://gemini.google.com/share/a80206554221
- https://www.youtube.com/watch?v=RQowiOF_FvQ&ist=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=8



