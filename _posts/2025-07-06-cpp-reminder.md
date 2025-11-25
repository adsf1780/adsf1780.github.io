---
title: C++ 기억할 것들
excerpt: C++ 쓸 때 유용하게 사용되는 것들
categories: [C++]
tags: [C++, tip]
pin: true
---

### 라이브러리
```cpp
#include <bits/stdc++.h> // 알고리즘 문제에 필요한 라이브러리 모음
```

### 입출력

```cpp
// 입출력 빠르게 하기

ios_base::sync_with_stdio(0);
cin.tie(0);

// 출력 값 소수점 설정
cout << fixed;
cout.precision(10); // 소수점 10자리 출력
```

```cpp
cout << endl; // 대신에
cout << '\n'; // 이거 쓰기. 훨씬 빠름
```

### Convert String to Integer(float, double...)

```cpp
#include <string>

// 정의
// int stoi (const string& str [, size_t* idx = 0, int base = 10]) : string to int

int integer = stoi("1234", nullptr, 10); // (문자열, nullptr, 진법) 진법의 기본값은 10
int integer = stoi("1234");

// stof, stod 등도 같은 방식으로 사용할 수 있음

```

### Convert Integer(double, long long...) to String

```cpp
#include <string>

// 정의
// string to_string(int val) : int(double, long long...) to string

string s = to_string(1234);

```

### Delimiter를 기준으로 split한 문자열 입력 받기

```cpp
#include <iostream>
#include <vector>
#include <sstream>
using namespace std;

vector<string> split(string str, char Delimiter){
    istringstream iss(str);
    string buffer;

    vector<string> result;

    while(getline(iss, buffer, Delimiter)){
        result.push_back(buffer);
    }

    return result;
}
```

### bitset 이용하여 2진수 형태로 출력하기

```cpp
#include <iostream>
#include <bitset>
using namespace std;

bitset<8> bit; // 00000000
bitset<8> bit("10101010"); // 문자열 "10101010"로 초기화
bitset<32> bit(127); // 10진수 127로 초기화

bit = 0 // 전체 0
bit = 1 // 전체 1
bit.flip() // 전체 반전

bit[3] = 0; // bit.reset(3);
bit[3] = 1; // bit.set(3, 1);
bit[3] = !bit[3]; // bit.flip(3);

string bit = bit.to_string(); // 문자열로 변환하여 반환

```
<a href="https://notepad96.tistory.com/35" class="btn btn--info">More Info</a>


### vector

```cpp
#include <vector>

// vector 정의
vector<int> v = {2, 4, 7, 9};

// 삭제
#include <algorithm> // remove 쓰기 위해 필요

v.erase(v.begin() + i); // i번째 인덱스의 원소 제거
v.erase(remove(v.begin(), v.end(), 3), v.end()); // 원소 7 제거(값이 7인 모든 요소를 제거)
v.erase(v.begin() + 1, v.begin() + 3); // 인덱스 1부터 인덱스 3까지의 원소 제거

// 할당
vector<int> v; // 함수 밖에서 정의
v.assign(n, 0); // 함수 안에서 n개의 0으로 초기화
v.resize(10); // 10만큼의 공간을 확보만 함. 즉 0~9까지의 인덱스로 접근 가능. reserve 쓰면 v[idx]가 UB임.

// 찾기
cout << find(v.begin(), v.end(), 7) - v.begin(); // 7의 index 반환

if(find(v.begin(), v.end(), 5) == v.end()){ // vector 내에 존재하지 않으면 v.end() 반환
    cout << "not exist";
}

// pair 찾기
vector<pair<int, int>> v;

auto it = find_if(v.begin(), v.end(), [=](const pair<int, int> p)-> bool{return p.first == elem1;});

// 모든 요소 더하기
#include <numeric>
int sum = accumulate(v.begin(), v.end(), 0); // 0은 합의 초기값

// pop
int last_element = v.back(); // vector의 마지막 원소 반환
v.pop_back(); // vector의 마지막 원소 제거. 반환 값 없음

// 정렬
#include <algorithm>

// 오름차순 정렬
sort(v.begin(), v.end()); 

// 내림차순 정렬
sort(v.rbegin(), v.rend()); 
sort(v.begin(), v.end(), greater<>());

bool compare(int a, int b){ // return 타입: bool
    return a > b; // 비교 연산자
}
sort(v.begin(), v.end(), compare); // false일 경우 swap

// pair가 요소일 때 초기화
vector<pair<int, int>> tree(n, {-1, -1}); // n개의 공간을 {-1, -1}로 초기화

// lower_bound
lower_bound(begin, end, value); // value 이상인 첫 번째 요소를 가리키는 iterator 반환. 정렬된 컨테이너에서만 동작. O(log n)
vector<int> v = {1, 2, 4, 4, 5, 6}; // 오름차순 정렬되어 있어야 함
auto it = lower_bound(v.begin(), v.end(), 4);
cout << "Index: " << (it - v.begin()); // 2

// 디버깅할 때 특정 인덱스까지만 출력하기
v._M_impl._M_start, 5 // 0-4번 인덱스만 보여줌

// 벡터 안의 모든 원소 제거
v.clear(); // capacity는 유지됨
```

### set

```cpp
#include <set> // #include <unordered_set>

set<int> s;

// 삽입
s.insert(1);
s.insert(3);
s.insert(7);

// 제거
s.erase(7); // 원소로 제거
s.erase(s.begin() + 1); // 1번째 인덱스 원소 제거
s.clear(); // set에 있는 모든 원소 삭제

// 찾기
s.find(3); // 원소 7에 해당하는 iterator 반환

// for문
for(set<int>::iterator it=s.begin(); it != s.end(); it++){
    cout << *it << '\n';
}

// 기타
s.count(1); // set 내에서 원소 1의 개수 반환
s.empty(); // 비어있으면 true, 아니면 false 반환
s.size(); // 세트에 들어있는 원소으 수 반환

// compare 구조체 정의해서 정렬 방식 커스텀하기
struct compare{ // const 안 붙이면 오류 남.
    bool operator()(const pair<int, int>& a, const pair<int, int>& b) const{
        if(a.second != b.second) return a.second > b.second;
        else return a.first > b.first;
    }
};

set<pair<int, int>, compare> list;

// 특징
// 1. 중복을 허용하지 않음
// 2. 삽입과 제거를 O(log(n))로 실행

// lower_bound
multiset<int> s = {1, 2, 4, 4, 5};
auto it = s.lower_bound(4); // 4 이상인 첫 번째 요소를 가리키는 iterator 반환. 정렬된 컨테이너에서만 동작. O(log n)


```

### map

```cpp
#include <map> // #include <unordred_map>

// map 정의
ordered_map<int, char> m;

// 삽입
m.insert({1, 'a'}); // {key, value}
m.insert({3, 'b'});
m.insert({6, 'c'});

// key로 value 얻기
cout << m[1] << endl; // 'a' 출력

// key로 삭제
m.erase(3); // key로 제거

// key로 탐색
cout << (m.find(1)) -> first << endl; // key 값 반환 = 1
cout << (m.find(1)) -> second << endl; // value 값 반환 = 'a'

if(m.find(2) == m.end()){
    cout << "Not Found" << endl; // key 값이 존재하지 않는 경우 m.end() 반환
}

// map의 value가 vector인 경우
unordered_map<int, vector<int>> tree;

tree[1].emplace_back(2);
tree[2].emplace_back(3);
tree[3].emplace_back(4);

// 쌍 개수
map<int, string> m;

m[1] = "apple";
m[2] = "banana";

m.size() // 2

// for문
// 1)  
for(auto &p : m){
    cout << "key: " << p.first << " value: " << p.second << endl;
}

// 2)
for(auto it = m.begin(); it != m.end(); it++){
    cout << "key: " << p->first << " value: " << p->second << endl;
}

// 3)
for(auto &[key, value] : m){
    cout << "key: " << key << " value: " << value << endl;
}
```

### queue
![](\assets\images\2025-07-06-cpp-reminder\image1.png)
```cpp
#include <queue>

// queue 정의
queue<int> q;

// 데이터 추가
q.push(element);

// 데이터 제거
q.pop(); // return 값 없음

// 제일 앞 데이터 반환
q.front();

// 제일 뒤 데이터 반환
q.back();

// 크기 반환
q.size();

// 비었는지 확인
q.empty(); 

// priority queue
priority_queue<int> pq; // max heap
pq.push(4);
pq.push(7);
pq.size();
pq.top();
pq.pop();

priority_queue<int, vector<int>, greater<int>> pq; // min heap
priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q; // min heap with pair

// compare 정의
struct compare{ // compare(a, b)가 true -> a는 b보다 우선순위 낮음 -> b가 top에 옴
    bool operator()(const T& a, const T& b){
        return a > b; 
    }
};

priority_queue<T, vector<T>, compare> pq;
```

### struct

```cpp
typedef struct Star{
	int u;
	int v;
	double w;
	Star(int u, int v, double w) : u(u), v(v), w(w) {} // 생성자 정의
} Star;
```

### pair

```cpp
#include <vector>

pair<int, int> a;
a=make_pair(2,4);
cout<< a.first;  // 2 출력
cout<< a.second; // 4 출력 

pair<int, double> p = make_pair(1, 2.1);

// for문에서 pair 쉽게 반복하기
vector<pair<string, int>> m;

for(auto& [key, value] : m){
    cout << key << " " << value << endl;
}
```

### 수학(절댓값, 반올림, 올림, 내림, 제곱, 제곱근)

```cpp
#include <cmath>

abs(-1); // 절댓값
double ceil(double x); // 올림(double, float, long double)
double floor(double x); // 내림(double, float, long double)
round(3.5); // 반올림

// 제곱
#include <cmath>

double pow(double base, double n); // float, long doulbe
double result = pow(2,10); // 2^10

// 제곱근
#include <cmath>

double sqrt(double x); // float, long double
double result = sqrt(4); // 2

// 유클리드 빗변
#include <cmath>

double hypotenuse = hypot(x, y); // sqrt(x^2 + y^2)

// 삼각함수
#include <cmath>
// sin, cos, tan acos, ...

// 10의 제곱의 과학적 표기
double x = 1e-6;  // 10^-6
```

### max(min), max_element(min_element)

```cpp
#include <algorithm>

int maxValue = max(1, 2); // 2

vector<int> v = {1, 2, 3, 4};
cout << *max_element(v.begin(), v.end()); // iterator 형태로 반환되므로 * 붙여야 함
cout << *min_element(v.begin(), v.end());
```

### string

```cpp
#include <string> // <iostream>에서 자동으로 include 되기는 하지만 명시적으로 포함하는 것이 좋음

// 빈 문자열 str 생성
string str; 

// 문자열 str 생성 방식 1
string str = "abc";

// 문자열 str 생성 방식 2
string str;
str = "abc";

// 문자열 str 생성 방식 3
string str("abc");

// str2에 str1 복사
string str2(str1);

// new를 이용해 동적할당
string *str = new string("abc");

// <, >, ==를 통해 문자열끼리 비교할 수 있다
string str1 = "abc";
string str2 = "bcd";
// str1 < str2 은 true
// str1 + "A" 은 "abcA"
// str1 == "abc" 는 true

// 특정 원소에 접근
str.at(index); // index 위치의 문자 반환. 유효한 범위인지 체크함
str[index]; // index 위치의 문자 반환. 유효한 범위인지 체크 안 함. 따라서 접근 더 빠름.
str.front(); // 문자열의 가장 앞 문자 반환
str.back(); // 문자열의 가장 뒤 문자 반환

// 추가, 삭제
str1.append(str2); // str1 뒤에 str2 이어 붙여줌('+'와 같은 역할)
str1.append(str2, n, m); // str1 뒤에 str2의 n번째 인덱스부터 m개의 문자를 이어 붙여줌
str.append(n, 'a'); // str 뒤에 n 개의 'a'를 이어 붙여줌
str1.insert(n, str2); // n번째 index 앞에 str2 문자열을 삽입함
str.clear(); // 저장된 문자열을 모두 지움
str.erase(n, m); // n번째 index부터 m개의 문자를 지움
str.pop_back(); // str 맨 뒤의 문자 제거

// 유용한 멤버 함수
str.find("abc"); // "abc"가 str에 포함되어있는지를 확인. 찾으면 해당 부분의 첫번째 index를 반환
str.find("abc", n); // 위와 비슷하지만, n번째 index부터 "abc"를 find함
str.substr(n); // n번째 index부터 끝까지의 문자를 부분문자열로 반환
str.substr(n, k); // n번째 index부터 k개의 문자를 부분문자열로 반환
swap(str1, str2); // str1과 str2를 바꿔줌. reference를 교환하는 방식
isdigit(c); // c가 숫자면 true, 아니면 false 반환
isalpha(c); // c가 영어면 true, 아니면 false 반환
toupper(c); // c를 대문자로 변환
tolower(c); // c를 소문자로 변환


// string의 크기
str.size(); // 문자열 길이 반환
str.capacity(); // 문자열이 사용 중인 메모리 크기 반환
str.resize(n); // str의 크기를 n으로 만듦. 기존의 문자열 길이가 n보다 크다면 초과하는 부분은 삭제하고, 작다면 빈공간으로 채움.
str.resize(n, 'a'); // 위와 비슷하지만, 빈 공간을 'a'로 채움.
str.shrink_to_fit(); // capacity가 실제 사용하는 메모리보다 큰 경우 낭비되는 메모리가 없도록 메모리를 줄여줌.
str.reserve(n); // str에 n만큼 메모리를 미리 할당해 줌.
str.empty(); // str이 빈 문자열인지 확인

// string 뒤집기
#include <algorithm>
reverse(str.begin(), str.end());
```

### class

```cpp
class Shark{
    // 멤버 변수
    public:
    int row;
    int col;
    int dist;

    // 생성자
    Shark(int row, int col, int dist) : row(row), col(col), dist(dist) {}
};
```

### iterator
```cpp
// 이하 백준 21939번 문제와 관련된 코드
#include <set>
#include <map>
#include <vector>

set<pair<int, int>> list;
map<int, set<pair<int, int>>::iterator> index; // set에 있는 요소를 빠르게 제거하기 위해서 pair의 first 값을 key로 하고, 그에 해당하는 set의 iterator를 value로 하는 map를 만듦.

// 요소 삽입
auto it = list.emplace(P, L).first; // set::emplace는 (iterator, bool) 반환.
index[P] = it;  // index 갱신 필요

// 요소 삭제
list.erase(index[P]);  // iterator를 통해 삭제. iterator도 삭제되기 때문에 이후에는 it 접근하면 UB
index.erase(P);        // index에서도 제거

// vector에서는 요소를 삭제하면 남아있는 요소들의 index가 변한다.
// 하지만 set에서는 요소를 제거해도 나머지 iterator들의 값이 여전히 유효하다.

// 이하 기타 내용
#include <iterator>

auto it = prev(set.end()); // set, map, vector 등의 마지막 iterator를 반환
```

### Dijkstra algorithm
```cpp
#include <vector>
#include <queue>
#include <climits>

vector<vector<pair<int, int>>> graph(n + 1); // graph[start].emplace(weight, end);

vector<int> dist(n+1, INT_MAX);
dist[k] = 0; // k: 시작 노드

// 최소힙을 써서 방문되지 않은 노드들 중 거리가 가장 짧은 노드의 이웃 노드를 방문함.
// top node는 최단 거리임이 보장됨. 왜냐하면 weight가 모두 양수이기 때문.
priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
q.emplace(0, k); // 시작 노드인 k의 distance는 0

while(q.size()){
    pair<int, int> current = q.top();
    q.pop();

    // visited된 노드라는 의미
    if(dist[current.second] < current.first) continue;

    for(pair<int, int> neighbor : graph[current.second]){
        if(current.first + neighbor.first < dist[neighbor.second]){
            dist[neighbor.second] = current.first + neighbor.first;
            q.emplace(dist[neighbor.second], neighbor.second);
        }
    }
}
```
<a href="https://velog.io/@panghyuk/%EC%B5%9C%EB%8B%A8-%EA%B2%BD%EB%A1%9C-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98" class="btn btn--info">More Info</a>

### binary search
```cpp
int binarySearch(vector<int>& arr, int target){
	int low = 0, high = arr.size() - 1;
    
    while(low < high){
    	int mid = (low + high) / 2;

        // mid는 내림으로 구해졌기 때문에 high = mid - 1을 하면 target에 정확히 도달할 수 없다.
        if(target > arr[mid]) low = mid + 1;

        // 원하는 값이 더 크다면 검사 범위를 더 크게 잡아야 한다.
        else high = mid;
    }
    return high; // return low 도 가능. low == high 일 때만 break 되기 때문
}

// 재귀적으로 구현
bool binarySearch(vector<int>& arr, int low, int high, int target){
    if(low > high)	return false;
    int mid = (low + high) / 2;
    
    if(arr[mid] == target)	return true;
    
    // 찾는 수가 더 작다면, 검사 범위를 더 작게 잡아야 한다.
    if(arr[mid] > target)
    	return binary_search(arr, low, mid - 1, target);
        
    // 찾는 수가 더 크가면, 검사 범위를 더 크게 잡아야 한다.
    else
    	return binary_search(arr, mid + 1, high, target);
}

// STL 이용
vector<int> nums;
int target = 3;
bool isFound = binary_search(nums.begin(), nums.end(), target);
// binary_search(반복자.시작점, 반복자.끝점, 찾고자 하는 값);
// 은 찾고자 하는 값을 찾으면 true를, 찾지 못하면 false를 반환한다.
```

### 테스트 케이스 개수 주어지지 않을 때

```cpp
int t;
while(cin >> t){
    cout << t << endl;
}
```

### 무한대

```cpp
// int 타입 무한대
#include <climits>

int inf = INT_MAX;

// float 타입 무한대
#incude <cmath>

float inf = INFINITY;
```

### 기타

- 1초에 약 1억 번($ =10^8 $) 연산
- $ int: 4byte = 32bit = -2^{31} \sim 2^{31} - 1 = -2,147,483,648 \sim 2,147,483,647  \approx 2 \times 10^9 $
- $ long long: 8byte = 64bit = -2^{63} \sim 2^{63}-1 \approx -9.22 \times 10^{18} \sim 9.22 \times 10^{18} $
- $ unsigned long long: 0 \sim 2^{64} - 1 \approx 10^{19} $