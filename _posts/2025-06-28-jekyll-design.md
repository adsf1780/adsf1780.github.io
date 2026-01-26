---
title: "jekyll 디자인 팁"
categories: [jekyll]
tags: [jekyll, post]
excerpt: jekyll로 블로그 작성할 때 팁들
---

## 이미지 추가하기

```markdown
![설명](\assets\images\2025-07-09-tree\image1.png)
```
<a href="https://mmistakes.github.io/minimal-mistakes/docs/helpers/#figure" class="btn btn--info">More Info</a>

## 이미지 크기 조정하기

<pre>![설명](경로)<strong>{: width="70%"}</strong></pre>

## 이미지 위치 조정하기

<pre>![설명](경로)<strong>{: .align-center}</strong></pre>

## 이미지 크기 및 위치 조절 한번에 하기

<pre>{: width="70%" .align-center}</pre>

## 이미지 나란히 삽입하기(2개, 3개 등...)

```html
<p align="center">
  <img src="#" width="40%">
  <img src="#" width="40%">
</p>
```

## 이미지에 caption 적용하기

```html
<p align="center"> 
  <img src="#">
  <figcaption>Caption</figcaption>
</p>
```

## 새로운 탭에서 열리도록 하이퍼링크 걸기

```html
<a href="#" target="_blank">#</a>
```

## 마크다운 적용 안 하기
```html
<pre></pre>
```
태그 활용하기

또는 

<pre>
```
```
</pre>
활용하기

## 일부 단락만 배경색 바꾸기(notice 활용)

```html
<div class="notice"> <!-- notice 대신 다른 class로 변경 가능 -->
  <h4 class="" style="font-size: 1.2em;">In JavaScript</h4><br> <!-- h4 태그만 제대로 작동함 -->
  shfit: remove the first element<br>
  push: add element at the end
</div>
```
<a href="https://mmistakes.github.io/minimal-mistakes/docs/utility-classes/#notices" class="btn btn--info">More Info</a>

## 버튼 생성하기

```html
<a href="경로" class="btn btn--info">More Info</a>
```
<a href="https://mmistakes.github.io/minimal-mistakes/docs/utility-classes/#buttons" class="btn btn--info">More Info</a>


## header image 삽입하기
![example image](\assets\images\2025-06-25-jekyll-remainder\mm-header-overlay-black-filter.jpg)
```yaml
excerpt: "This post should [...]"
header:
  overlay_image: /assets/images/unsplash-image-1.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)" # 이미지 우측 하단에 표시됨
  actions:
    - label: "Download"
      url: "https://github.com"
```
<a href="https://mmistakes.github.io/minimal-mistakes/docs/layouts/#header-overlay" class="btn btn--info">More Info</a>

## 화살표 넣기

```markdown
왼쪽 화살표: &larr;
오른쪽 화살표: &rarr;
```

## 내용 숨기기

category 전체 숨기기: data/navigation.yml 에서 all_posts의 hide에 숨기고 싶은 category를 추가한다.

특정 post만 숨기기: 해당 post의 Front Matter에서 show_each_post: false 를 입력한다.

## 수식 삽입하기

Front Matter에 `use_math: true` 넣기

- inline 방식으로 수식을 삽입할 때
```markdown
$ 수식 $
```

- block 방식으로 수식을 삽입할 때
```markdown
$$
수식
$$
```

- 띄어쓰기(작게)
```markdown
$ \, $
```

- 띄어쓰기(크게)
```markdown
$ \; $
```

- 줄 변환
```markdown
$ \\ $
```

- 분수
```markdown
$ \frac{분자}{분모} $
```


<a href="https://yoongyoonge.github.io/blog-mathematical-expression/" class="btn btn--info">More Info</a>
