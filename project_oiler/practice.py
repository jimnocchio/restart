#출력값: 2^1000의 각 자릿수를 모두 더하면?
#풀이: 우선 포문으로 계싼을하고 그걸 리스트로 만들어서 각 리스트끼리 덧셈을한다.

# result=0
# num=str(2**1000)
# for i in num[0:]:
#     result+=int(i)
# print(result)
    
#다음수중에 더 큰걸 고르는거는 1차원적인 사고
#다다음 수까지 보면서 계산해야할듯
#리스트로 만들어서 계산?

#100!
#리스트로 확인후 더하기.

result=0
def factorial(num):
    if num==0:
        return 1
    else:
        return num*factorial(num-1)
n=str(factorial(100))
for i in n:
    result+=int(i)
print(result)