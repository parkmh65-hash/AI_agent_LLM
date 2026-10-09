from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

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
    history: List[Dict[str, str]] = []

def get_article_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 콜랩 예제에 명시된 특정 영역 추출 로직
        article = soup.find('article', class_='story-news article')
        if article: return article.get_text(strip=True)
        
        if soup.find('article'):
            return soup.find('article').get_text(strip=True)
        
        cm_ad = soup.find('div', id="CmAdContent")
        if cm_ad:
            return cm_ad.get_text(strip=True)
            
        return "기사 내용을 찾을 수 없습니다."
    except Exception as e:
        return f"URL을 가져오는 중 오류 발생: {e}"

def get_search_context(query: str) -> str:
    # 1. DuckDuckGo API wrapper (한국지역, 최근 일주일)
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time="w")
    
    # 2. DuckDuckGoSearchResults (뉴스 소스)
    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper, 
        source="news", 
        results_separator=';\n'
    )
    
    # 검색 실행 (에러 방지를 위해 try-except 추가)
    try:
        # 특정 사이트(ytn.co.kr)를 명시하거나 질문 자체를 검색
        # ytn 검색을 원할 경우 query 앞에 "site:ytn.co.kr "를 붙일 수 있습니다.
        docs = search.invoke(query)
    except Exception as e:
        return "검색 중 오류가 발생했습니다."
    
    # 3. 링크 추출 및 스크래핑
    links = []
    if docs:
        for doc in docs.split(";\n"):
            if "link:" in doc:
                link = doc.split("link:")[1].strip()
                links.append(link)
    
    articles = []
    # 토큰 제한을 막기 위해 상위 2개의 기사만 스크래핑
    for link in links[:2]:
        text = get_article_text(link)
        # 내용이 너무 길면 자름
        articles.append(f"[출처: {link}]\n{text[:1500]}")
    
    return "\n\n".join(articles) if articles else "관련 기사를 찾을 수 없습니다."

@app.post("/api/chat")
def chat_with_bot(req: ChatRequest):
    try:
        # LLM 초기화
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
        
        # LCEL 프롬프트 설정
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{question}")
        ])
        
        # 체인 구성
        chain = prompt | llm | StrOutputParser()
        
        # 웹 스크래핑을 통한 컨텍스트 수집
        context = get_search_context(req.message)
        
        # 메모리(히스토리) 포맷팅
        formatted_history = []
        for h in req.history:
            formatted_history.append((h["role"], h["content"]))
            
        # 결과 생성 (에러가 발생하기 가장 쉬운 구간)
        result = chain.invoke({
            "context": context,
            "messages": formatted_history,
            "question": req.message
        })
        
        return {"reply": result}
        
    except Exception as e:
        # 500 에러 대신 200 상태 코드로 상세 에러 메시지를 프론트로 반환
        return {"reply": f"⚠️ 서버 처리 중 오류가 발생했습니다: {str(e)}"}
