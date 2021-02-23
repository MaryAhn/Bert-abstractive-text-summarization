from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from konlpy.tag import Mecab
import re
import os
import pandas as pd

mecab = Mecab()

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    cnt = 0

    for _ in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=True):
        cnt += 1

    pagenos = set(range(1, cnt))

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)
        contents = retstr.getvalue()
        contents.encode('utf-8')

    fp.close()
    device.close()
    retstr.close()
    return contents

def spacing_mecab(wrongSentence):
    tagged = mecab.pos(wrongSentence)
    corrected = ""
    for i in tagged:
        # print(i[0], i[1])

        if i[1] in ('JKS', 'JKC', 'JKG', 'JKO', 'JKB' ,'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'SA', 'SF'):
            corrected += i[0]
        else:
            corrected += " "+i[0]

    if not corrected:
        return corrected+'.'
    if corrected[0] == " ":
        corrected = corrected[1:]
    return corrected

def clean_sentense(txt):
    pattern = '(\d\d\d-\d\d\d\d-\d\d\d\d)'  # 전화번호 제거 (000-0000-0000),\d: 숫자 1개
    txt = re.sub(pattern=pattern, repl='', string=txt)
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'  # E-mail제거, a-z 사이의 문자,
    txt = re.sub(pattern=pattern, repl='', string=txt)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # URL제거
    txt = re.sub(pattern=pattern, repl='', string=txt)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    txt = re.sub(pattern=pattern, repl='', string=txt)
    pattern = '<[^>]*>'  # HTML 태그 제거
    txt = re.sub(pattern=pattern, repl='', string=txt)
    # pattern = '[^\w\s]'  # 특수기호제거
    # txt = re.sub(pattern=pattern, repl='', string=txt)
    pattern = '([一-龥])' # 한자 제거
    txt = re.sub(pattern=pattern, repl='', string=txt)


    return txt


path_open = '/Users/angeonhui/Bert-abstractive-text-summarization/data/dataset/train/data_pdf'
path_save = '/Users/angeonhui/Bert-abstractive-text-summarization/data/dataset/train/data_csv/philosophy.csv'
text_name_list = [os.path.splitext(f)[0] for f in os.listdir(path_open) if f.lower().endswith('.pdf')]
pdf_df = []

for pdf in text_name_list:
    path = os.path.join(path_open, ('%s.pdf' % (pdf)))
    text = convert_pdf_to_txt(path)

    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/05 15:25(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/06 14:24(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/06 14:21(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/06 14:22(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/06 14:23(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/08 13:50(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/08 13:51(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/08 13:52(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/08 13:53(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/08 13:54(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/08 13:46(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/08 13:47(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/08 13:48(KST)', '')
    text = text.replace('이화여자대학교 | IP:203.255.***.68 | Accessed 2021/02/08 13:49(KST)', '')

    extracted = re.split('논문개요|주제어|참고문헌', text)

    for i in range(len(extracted)):
        extracted[i] = clean_sentense(extracted[i])
        extracted[i] = spacing_mecab(extracted[i])
    # print(extracted)

    text_dic = {}
    text_dic['title'] = extracted[0]
    text_dic['summary'] = extracted[1]
    text_dic['content'] = extracted[2]
    # print(text_dic)

    paper_df = pd.DataFrame(text_dic, index=[pdf])
    pdf_df.append(paper_df)

new_text = pd.concat(pdf_df, axis=0).reset_index(drop=True)
new_text.to_csv(path_save)







