#출력값: 2^1000의 각 자릿수를 모두 더하면?
#풀이: 우선 포문으로 계싼을하고 그걸 리스트로 만들어서 각 리스트끼리 덧셈을한다.

result=0
num=str(2**1000)
for i in num[0:]:
    result+=int(i)
print(result)
    
