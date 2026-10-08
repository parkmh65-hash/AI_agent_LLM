from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# GAS에서 API를 호출할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewData(BaseModel):
    review: str

@app.post("/api/analyze")
def analyze_review(data: ReviewData):
    # 모델 초기화 (환경 변수 OPENAI_API_KEY 자동 참조)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    parser = StrOutputParser()

    # 1. 리뷰 요약
    prompt1 = PromptTemplate.from_template("다음 식당 리뷰를 한 문장으로 요약하세요.\n{review}")
    chain1 = prompt1 | model | parser
    summary = chain1.invoke({"review": data.review})

    # 2. 긍정/부정 점수
    prompt2 = PromptTemplate.from_template("다음 식당 리뷰를 읽고 0점부터 10점 사이에서 긍정/부정 점수를 매기세요. 숫자만 대답하세요.\n{review}")
    chain2 = prompt2 | model | parser
    sentiment_score = chain2.invoke({"review": data.review})

    # 3. 답변 작성
    prompt3 = PromptTemplate.from_template("다음 식당 리뷰 요약에 대해 고객에게 공손한 답변을 작성하세요.\n리뷰 요약:\n{summary}")
    chain3 = prompt3 | model | parser
    reply = chain3.invoke({"summary": summary})

    return {
        "summary": summary,
        "sentiment_score": sentiment_score,
        "reply": reply
    }
