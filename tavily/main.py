from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import uvicorn  # 필수: 맨 하단 서버 실행을 위한 모듈

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/api/chat")
def generate_report(req: ChatRequest):
    try:
        # 1. Tavily 검색 실행
        tavily = TavilyClient()
        search_res = tavily.search(
            req.message,
            search_depth="advanced", 
            include_raw_content=True
        )
        context = search_res.get("results", [])
        
        # 2. 제미나이 LLM 초기화 (쉼표 누락 및 모델명 오타 수정)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.2,
            max_retries=3  # 429 에러 시 최대 3회 자동 재시도
        )

        # 3. 프롬프트 설정
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 신문기사를 쓰는 기자 AI입니다. 당신은 주어진 정보를 바탕으로 객관적이고 체계적으로 작성된 기사를 써야 합니다."),
            ("user", "정보: \"\"\"{context}\"\"\"\n\n위의 정보를 사용하여, 다음 질문에 대해 자세한 보고서를 한국어로 작성하세요: \"{query}\"\n—신문기사 형식을 사용하되, MLA를 준수하는 markdown 문법을 사용해주세요.\n—활용한 자료는 출처를 명시하세요.")
        ])
        
        # 4. LCEL 체인 구성 및 실행
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({
            "context": context,
            "query": req.message
        })
        
        return {"reply": result}
        
    except Exception as e:
        error_msg = str(e)
        # 429 할당량 초과 에러인 경우 사용자 친화적 메시지 반환
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {"reply": "⏳ 현재 이용자가 많아 AI 생성 한도를 초과했습니다. 약 20초 후에 [작성] 버튼을 다시 눌러주세요."}
        
        return {"reply": f"⚠️ 텍스트 생성 중 오류가 발생했습니다: {error_msg}"}

if __name__ == "__main__":
    # Cloud Run이 제공하는 PORT 환경변수를 가져오되, 없으면 8080 사용
    port = int(os.environ.get("PORT", 8080))
    # 외부 접속이 가능하도록 host를 "0.0.0.0"으로 설정
    uvicorn.run(app, host="0.0.0.0", port=port)
