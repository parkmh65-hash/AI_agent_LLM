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
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """다음 영상에 대한 요약을 한국어로 만들어줘:
            {context}""")
        ])
        chain = create_stuff_documents_chain(llm, prompt)

        results = []
        for v in videos:
            v_url = 'https://youtube.com' + v['url_suffix']
            try:
                # 1. 객체 생성() 없이 클래스 메서드 get_transcript 직접 호출
                fetched = YouTubeTranscriptApi.get_transcript(v['id'], languages=['ko', 'en'])
                
                # 2. 반환값이 딕셔너리 리스트이므로 s['text']로 키에 접근
                text = " ".join([s['text'] for s in fetched])
                
                # 3. LangChain 체인에 전달하기 위해 Document 객체로 매핑
                docs = [Document(page_content=text)]
                
                summary = chain.invoke({"context": docs})
                
            except Exception as e:
                summary = f"자막 추출 오류: {type(e).__name__}: {e}"
                
            results.append({
                "title": v.get("title"),
                "url": v_url,
                "duration": v.get("duration"),
                "summary": summary
            })

        return {"data": results}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Cloud Run이 제공하는 PORT 환경변수를 가져오되, 없으면 8080 사용
    port = int(os.environ.get("PORT", 8080))
    # 외부 접속이 가능하도록 host를 "0.0.0.0"으로 설정
    uvicorn.run(app, host="0.0.0.0", port=port)
