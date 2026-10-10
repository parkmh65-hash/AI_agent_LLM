from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_search import YoutubeSearch
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import uvicorn
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

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
                fetched = YouTubeTranscriptApi.get_transcript(v['id'], languages=['ko', 'en'])
                text = " ".join([s['text'] for s in fetched])
                
                docs = [Document(page_content=text)]
                summary = chain.invoke({"context": docs})
                
            except (TranscriptsDisabled, NoTranscriptFound):
                # 자막이 아예 없거나, 요청한 언어(ko, en) 자막이 없는 경우
                summary = "이 영상은 자막이 제공되지 않아 요약할 수 없습니다."
            except Exception as e:
                # 그 외의 알 수 없는 오류
                summary = f"자막 추출 오류: {type(e).__name__}"
            
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
