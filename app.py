from fastapi import Body, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, constr
from pyserini.search.lucene import LuceneSearcher
from pyserini.index import IndexReader
from gensim import models
import re
import json
import random
import uvicorn
import CH_HLSQG

# 載入 中文QGmodel、w2v model
ch_model = CH_HLSQG.BERT_HLSQG()
w2v_model = models.Word2Vec.load('/user_data/mcq/w2v_zh/word2vec_wiki_zh.model.bin')
# 載入 Inverted Table
print('Loading Inverted Index for Wikipedia...')
searcher = LuceneSearcher('Inverted_Index_Wiki_Chinese')
index_reader = IndexReader('Inverted_Index_Wiki_Chinese')

# 定義Retreiver 使用從corpus建好的Inverted_Index_Wiki_Chinese
def Retreiver(quesion):
  searcher.set_language('zh')
  hits = searcher.search(quesion, k=10)
  return hits

# 答案選項生成
def answerPosFinder(cxt, ans):
  start = cxt.find(ans)
  end = start+len(ans)
  ans_pos = [(m.start(), m.end()) for m in re.finditer(ans, cxt)]

  QGpairs=[]
  for i in ans_pos:
    pair = {"article":cxt,
            "answers":{
                "ans_detail":[
                  {
                      "tag":ans,
                      "start_at":i[0],
                      "end_at":i[1]
                  }            
                ]
            }
            }
    QGpairs.append(pair)
      
  return QGpairs

# 定義w2v
def word2vec_predict(w2v_model,predictWord,top_k=10,verbose=False):
  try:
    rt = []
    predict = w2v_model.wv.similar_by_word(predictWord, topn=top_k)
    for word in predict:
      rt.append(word[0])
    return rt
    
  except:
    print('字典裡沒這個word: {}!'.format(predictWord))
    return 'error'

# 從網頁拿取資料的format
class MCQRequest(BaseModel):
    maintext: constr(max_length=512)

# build FastAPI
app = FastAPI(
    title="Multiple Zhoice Question Generation",
    description="Multiple Choice Question Generation",
    version="0.1.0",
)

# mount js、html
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 定義root 目前直接導到mcq.html
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("mcq.html",{"request": request})

# mcq
@app.post("/mcq")
async def mcq(
    mcq_request: MCQRequest = Body(
        None,
    )
):
    print(mcq_request)
    # 從前端取得使用者下達的關鍵字詞。
    term_from_front_end_user = mcq_request.maintext 
    topics = str(term_from_front_end_user)
    # 目前先用topic關鍵字的第一個詞來當答案。(下面是隨機選擇答案)
    QG_ans=topics.split()[0] 
    QG_ans=random.choice(topics.split())
    print(QG_ans)

    # 定義question set
    q_set=set()
    # 藉由topics(使用者的輸入)去找相關documents
    hits = Retreiver(topics)
    # 
    for i in hits:
      # 拿取Retreiver回傳的document 裡的contents
      contents = json.loads(i.raw)["contents"]
      print(contents)
      # 從contents中找到答案(QG_ans)的位置並回傳qa(問題 答案位置)
      for qa in answerPosFinder(contents,QG_ans)[:5]:
        context = qa["article"]
        answer = qa["answers"]["ans_detail"][0]['tag']
        answer_start = qa["answers"]["ans_detail"][0]['start_at']
        # 找文章長度大於100的
        if len(context) >= 100:
          # 用context尋找問題 
          questions = ch_model.predict(context = context, answer = answer, answer_start = answer_start, BS = 1)
          for q in questions:
            # 判斷QG_ans是否在question裡面 找沒有的且長度大於12
            if not QG_ans in q:
              if len(q)>12:
                q_set.add(q)

    # 所有的question
    question_set = list(q_set)
    #處理[UNK]
    for index,q in enumerate(question_set):
      if "[UNK]" in q:
        question_set[index].replace("[UNK]","")

    question_set_len=len(question_set)
    choice_output = []
    # a_dict=[]
    # for i in question_set:
    #   a_dict.append(QG_ans)

    for k in range(question_set_len):
      top_k=5
      verbose=True
      # 相關選項list
      distrator = word2vec_predict(w2v_model,QG_ans,top_k,verbose)
      #print(distrator)
      if not distrator == 'error':
        for a in distrator:
          if QG_ans in a: 
            distrator.remove(a)
        # **CODE** 判斷distrator是否長度過短

        # 取前十個distrator來打亂隨機選擇
        distrator_10=distrator[:10]
        random.shuffle(distrator_10)
        ans = [QG_ans]
        ans.append(distrator_10[-3])
        ans.append(distrator_10[-2])
        ans.append(distrator_10[-1])
        random.shuffle(ans)
        # 例子 A:白血病 B:乳腺癌 C:腎病 D:前列腺癌
        choice_output.append(
              "A:" +ans[0]+"  "+
              "B:" +ans[1]+"  "+
              "C:" +ans[2]+"  "+
              "D:" +ans[3]+"  " 
        )
      else:
        return ["查不到這個詞彙，請重新搜尋 :「"+QG_ans+"」"]
    # 看選項有什麼
    for i in choice_output:
        print("選項:",i)

    # 最後要return的list 下面例子
    # Q1: 髓細胞會因為哪一種病變而發生異常的生長?
    # A:白血病 B:乳腺癌 C:腎病 D:前列腺癌
    question_set_ans=[]
    
    # 結合question_set choice_output
    for i, question in enumerate(question_set):
      question_set_ans.append("Q"+str(i+1)+": "+question)
      question_set_ans.append(choice_output[i])
    
    print("***********")
    print(question_set_ans[:12])
    print("***********")
    return question_set_ans[:12]

if __name__ == '__main__':
    uvicorn.run(app= 'app:app',host='0.0.0.0',port=16208,reload=True)