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

class SearchQuery(BaseModel):
    query: str

@app.post("/api/report")
def generate_report(req: SearchQuery):
    try:
        # 1. Tavily 검색 실행 (환경 변수 TAVILY_API_KEY 자동 참조)
        tavily = TavilyClient()
        search_res = tavily.search(
            req.query, 
            search_depth="advanced", 
            include_raw_content=True
        )
        context = search_res.get("results", [])
    except Exception as e:
        return {"report": f"Tavily 검색 중 오류가 발생했습니다: {str(e)}"}

    # 2. 제미나이 LLM 초기화 (환경 변수 GOOGLE_API_KEY 자동 참조)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-lates", temperature=0.2)
    
    # 3. 프롬프트 설정 (콜랩 소스 기반)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 신문기사를 쓰는 기자 AI입니다. 당신은 주어진 정보를 바탕으로 객관적이고 체계적으로 작성된 기사를 써야 합니다."),
        ("user", "정보: \"\"\"{context}\"\"\"\n\n위의 정보를 사용하여, 다음 질문에 대해 자세한 보고서를 한국어로 작성하세요: \"{query}\"\n—신문기사 형식을 사용하되, MLA를 준수하는 markdown 문법을 사용해주세요.\n—활용한 자료는 출처를 명시하세요.")
    ])
    
    # 4. LCEL 체인 구성 및 실행
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "context": context,
        "query": req.query
    })
    
    return {"report": result}
