import numpy as np

class Embedding:
    def __init__(self,W):
        #해당 임베딩 계층에 전달된 가중치 SET을 저장한다.
        self.params = [W]
        #해당 계층의 매개변수의 기울기들의 형상은 기울기 셋이랑 동일하다
        #그래서 가중치 set을 0으로 초기화 한 뒤에 형상을 저장한다.
        self.grads =[np.zeros_like(W)]
        #특정 임배딩 계층에서 추출할 가중치의 행 번호
        #(가중치가 의미 하는 것은 은닉 뉴런들의 SET이다) 
        self.idx = None
    #임베딩 계층의 순전파는 은닉 뉴런들을 즉, 가중치 셋의 하나의 행을 출력하는 것
    def forward(self,idx):
        # W 뒤에 쉼표를 넣는 것은 패킹을 푼다는 거 (생성자에서 배열로 만들라고 패킹됨 ㅋㅋ)
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out 
        #순전파는 별 거 없군 !
    
    #dout은 앞의 계층으로부터 전달 받은 손실 변화량이다.
    def backward(self,dout):
        #생성자 때의 패킹 풀기 -> 근데 생각해 보니까 처음 생성자 때 패킹 풀고 받으면 안되나?
        #두번 풀어야 되는데
        dW, = self.grads
        dW[...] = 0

        #변수가 뽑히는 시점의 인덱스와 해당 인덱스의 값이 같이 들어감 = enumerate
        for i, word_id in enumerate(self.idx):
            #해당 가중치의 행을 
            dW[word_id] +=dout[i]





