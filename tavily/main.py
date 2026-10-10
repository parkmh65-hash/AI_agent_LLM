from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프론트엔드(Code.gs)에서 'message'와 'history'를 보내므로 규격을 맞춰줍니다.
class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/api/chat")
def generate_report(req: ChatRequest):
    try:
        # 1. Tavily 검색 실행
        tavily = TavilyClient()
        search_res = tavily.search(
            req.message, # 프론트에서 넘어오는 키(message) 사용
            search_depth="advanced", 
            include_raw_content=True
        )
        context = search_res.get("results", [])
        
        # 2. 제미나이 LLM 초기화 (오타 수정됨)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
        
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
        
        # 프론트엔드에서 기다리는 'reply' 키로 반환
        return {"reply": result}
        
    except Exception as e:
        # 서버 다운 대신 챗봇 화면에 에러 원인 출력
        return {"reply": f"⚠️ 텍스트 생성 중 오류가 발생했습니다: {str(e)}"}
