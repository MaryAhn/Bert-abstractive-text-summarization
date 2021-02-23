from konlpy.tag import Mecab
tokenizer = Mecab()

txt_file = open("/Users/angeonhui/Bert-abstractive-text-summarization/data/dataset/for_vocab/all_text_0216.txt", 'r')
text_data = txt_file.read()
txt_file.close()

def whitespace_tokenize(data):
  data = data.strip()    # 문자열의 맨앞, 맨끝 공백 지움
  if not data:
    return []
  tokens = data.split()  # 문자열을 스페이스,탭,엔터 단위로 분리하여 배열에 집어넣음
  return tokens

output_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

for wst in whitespace_tokenize(text_data):  # wst : 공백,탭,엔터 기준 문자열 하나
    count = 0
    for token in tokenizer.morphs(wst):  # token : wst를 형태소 분석한 토큰 하나
        tk = token

        if count > 0:
            tk = "##" + tk
            if tk in output_tokens:  # 토큰이 중복되면 저장하지 않음
                continue
            output_tokens.append(tk)
        else:  # count==0
            count += 1
            if tk in output_tokens:  # 토큰이 중복되면 저장하지 않음
                continue
            output_tokens.append(tk)  # 맨 처음 token만 앞에 ##을 붙이지 않음

vocab_file = open('/Users/angeonhui/Bert-abstractive-text-summarization/data/dataset/for_vocab/vocab_0216.txt', 'w')
cnt = 0
for token in output_tokens:
  vocab_file.write(token + '\n')
  cnt += 1

vocab_file.close()
print('total vocab:', cnt)
