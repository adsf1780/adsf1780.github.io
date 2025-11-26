---
title: "jekyll 기억할 것들"
header:
  overlay_image: https://stqnq5ux4599.edge.naverncp.com/data2/content/image/2023/04/24/.cache/512/202304240914899.jpg
categories: [jekyll]
tags: [jekyll, post]
excerpt: jekyll로 블로그 작성하기 위해 기억할 내용들
show_each_post: false
---

## Server Local에서 Host하기

1. PowerShell에서 asdf178.github.io 폴더로 경로 이동하기
2. bundle exec jekyll s 또는 bundle exec jekyll serve (- -livereload) 입력하기
3. http://127.0.0.1:4000/ 열기

## Post 작성하기

1. _posts 폴더 아래에 년도-월-일-제목.md 형식의 파일 만들기

2. Front Matter에 title, categories, tags, excerpt와 같은 기본 정보 설정하기

   ```yaml
   ---
   title: 제목
   categories: [Top_Category, Sub_Category] # up to 2 elements
   tags: [tag1, tag2, ...] # TAG names should always be lowercase. 0~INFINITY
   excerpt: Short summary of the post. # or the first words of the post are used.
   typora-root-url: ../
   sidebar:
     nav: "docs"
   date: YYYY-MM-DD HH:MM:SS
   toc: false # If you want to turn off TOC(Table of Contents) for a specific post, add this line.
   pin: true # Pin one or more posts to the top of the home page 
   header: # 미리보기 이미지
     teaser: /assets/images/my-awesome-post-teaser.jpg 
   searh: false # Can't find this blog through search
   use_math: true # Latex 기능 쓰기
   ---
   ```

3. 내용 작성하기

## author 추가하기

1. _data/authors.yml에 새로운 author 정보 추가하기

2. Front Matter의 author에 작성자 이름(KSE) 적기

   ```yaml
   ---
   author: <author_id> # for single entry
   # or
   authors: [<author1_id>, <author2_id>]   # for multiple entries
   ---
   ```
   

## Latex로 수식 쓰기

1. Front Matter에 `use_math: true` 넣기
2. `$ 수식 $`  형식으로 .md 파일에 적기


## style 변경 사항 찾기
**Ctrl + Shift + F &rarr; 전체 파일에서 검색**으로 **조정** 검색<br>
(왜냐하면 css 파일을 변경할 때 주석에 '조정'이라고 써놓음)