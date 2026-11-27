import numpy as np

#말뭉치를 단어 사전(id_to_word,word_to_id로 변환 한 뒤 말뭉치 배열 corpus 반환)
def preprocess(text):
    text = text.lower()
    #공백 기준으로 단어를 구분하기 위함
    text = text.replace('.',' .')
    words = text.split(" ")
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

#말뭉치로부터 맥락과 타깃을 생성
def create_contexts_target(corpus,window_size = 1):
    target = corpus[window_size:-window_size]
    contexts =[]

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size,window_size+1):
            if t==0:
                continue
            cs.append(corpus[idx+t])

        contexts.append(cs)
        return np.array(contexts),np.array(target)

#말뭉치 자체를 원핫으로 변경
def convert_one_hot(corpus, vocab_size):
    corpus = np.asarray(corpus)
    return np.eye(vocab_size, dtype=np.int32)[corpus]



#맥락과 타깃을 원 핫으로 변경하는 함수 , vocab_size = len(word_to_id)
def create_one_hot_contexts_target(corpus, vocab_size, window_size=1):
    contexts, target = create_contexts_target(corpus, window_size)

    contexts = convert_one_hot(contexts, vocab_size)
    target = convert_one_hot(target, vocab_size)

    return contexts, target