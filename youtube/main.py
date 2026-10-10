from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_search import YoutubeSearch
from langchain_community.document_loaders import YoutubeLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
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

@app.post("/api/youtube")
def summarize_youtube(req: SearchQuery):
    try:
        # 1. 유튜브 검색
        videos = YoutubeSearch(req.query, max_results=3).to_dict()
        
        # 2. 60분 이하 영상 필터링
        videos = [v for v in videos if len(v['duration'].split(':')) < 3]

        # 3. 모델 및 체인 구성 (GOOGLE_API_KEY 환경변수 참조)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """다음 영상에 대한 요약을 한국어로 만들어줘:

{context}""")
        ])
        chain = create_stuff_documents_chain(llm, prompt)

        results = []
        for v in videos:
            v_url = 'https://youtube.com' + v['url_suffix']
            try:
                # 자막 로드
                loader = YoutubeLoader.from_youtube_url(v_url, language=['ko', 'en'])
                docs = loader.load()
                
                if docs:
                    summary = chain.invoke({"context": docs})
                else:
                    summary = "자막을 제공하지 않는 영상입니다."
            except Exception as e:
                summary = f"자막 추출 오류: {str(e)}"
            
            results.append({
                "title": v.get("title"),
                "url": v_url,
                "duration": v.get("duration"),
                "summary": summary
            })

        return {"data": results}
    except Exception as e:
        return {"error": str(e)}
