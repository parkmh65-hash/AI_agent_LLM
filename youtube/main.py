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
                # 1. 자막 리스트를 먼저 가져옴
                transcript_list = YouTubeTranscriptApi.list_transcripts(v['id'])
                
                # 2. 한국어, 영어, 자동 생성 영어 중 가능한 것을 찾음
                try:
                    # 먼저 한국어 또는 표준 영어를 시도
                    transcript = transcript_list.find_transcript(['ko', 'en'])
                except NoTranscriptFound:
                    # 없으면 사용 가능한 아무 자막(주로 자동 생성된 자막)을 영어로 번역해서 시도
                    transcript = transcript_list.find_generated_transcript(['en', 'ko'])
                    # transcript = transcript.translate('ko') # 필요하다면 여기서 한국어로 번역 요청 가능

                fetched = transcript.fetch()
                text = " ".join([s['text'] for s in fetched])
                
                docs = [Document(page_content=text)]
                summary = chain.invoke({"context": docs})
                
            except (TranscriptsDisabled, NoTranscriptFound):
                summary = "이 영상은 자막이 제공되지 않아 요약할 수 없습니다."
            except Exception as e:
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
