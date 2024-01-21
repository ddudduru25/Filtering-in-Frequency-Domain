# Filtering in Frequency Domain

## Digital Image Processing - week 5

### Frequency Domain Analysis
* 영상 신호를 주파수 영역에서 분석
* Fourier transform을 통해 변환
* Magnitude 성분과 phase 성분으로 나누어 시각화 가능
* 주파수 영역에서는 주기적으로 반복되는 성분을 쉽게 파악 가능

### 기능
1. LPF(Low pass filtering)
   * 저주파 성분만 통과시키고 고주파 성분은 배제하는 필터링
   * 값이 크게 바뀌는 영역에서 고주파 성분을 차단하여 영상 smoothing 효과 가능
2. HPF(High pass filtering)
   * 고주파 성분만 통과시키고 저주파 성분은 배제하는 필터링
   * 값이 크게 바뀌는 영역에서 고주파 성분만 남겨 영상 edge 추출 가능

