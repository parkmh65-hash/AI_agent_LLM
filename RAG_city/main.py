import os
import json
import tempfile
import traceback
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict

LangChain 및 AI 모델 관련 모듈

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

Supabase 클라이언트 모듈

from supabase import create_client, Client

app = FastAPI()

클라이언트(GAS) 통신을 위한 CORS 설정

app.add_middleware(
CORSMiddleware,
allow_origins=[""],
allow_credentials=True,
allow_methods=[""],
allow_headers=["*"],
)

환경 변수 로드 (에러를 세밀하게 잡기 위해 명시적 확인)

try:
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
except KeyError as e:
print(f"[환경변수 누락 오류] 서버 구동에 필요한 환경변수가 없습니다: {e}")

API 클라이언트 초기화

try:
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# 1. 질의 확장용 모델 (gemini-3.5-flash-lite, 빠른 처리)
llm_fast = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY)

# 2. 답변 생성용 스트리밍 모델 (gemini-3.5-flash-lite)
llm_streaming = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY, streaming=True)


except Exception as e:
print(f"[초기화 오류] API 클라이언트 초기화 실패: {e}")

class ChatRequest(BaseModel):
query: str
history: List[Dict[str, str]] = [] # [{"role": "user", "content": "..."}, ...]

@app.get("/health")
async def health_check():
"""서버 구동 상태 확인 엔드포인트"""
return {"status": "ok", "message": "Server is running normally."}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
"""PDF 파일을 업로드받아 텍스트 추출 후 청크 분할 및 Supabase 벡터 DB 저장"""
try:
# 임시 파일로 PDF 저장
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
tmp.write(await file.read())
tmp_path = tmp.name

    # PDF 파싱
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        raise Exception(f"PDF 파싱 중 오류가 발생했습니다: {str(e)}")
    
    # 텍스트 청크 분할
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
    except Exception as e:
        raise Exception(f"텍스트 분할(Chunking) 중 오류가 발생했습니다: {str(e)}")

    # 임베딩 및 벡터 DB 저장
    try:
        for chunk in chunks:
            meta = chunk.metadata
            meta["source_file"] = file.filename
            # 오픈 AI 임베딩 모델 사용
            vector = embeddings.embed_query(chunk.page_content)
            
            # Supabase에 RAG 데이터 삽입
            supabase.table("documents").insert({
                "content": chunk.page_content,
                "metadata": meta,
                "embedding": vector
            }).execute()
    except Exception as e:
        raise Exception(f"Supabase 벡터 DB 임베딩 및 저장 중 오류가 발생했습니다: {str(e)}")

    os.remove(tmp_path)
    return {
        "filename": file.filename, 
        "text": full_text[:1000] + "...(생략)", # 프리뷰 제공
        "message": f"총 {len(chunks)}개의 청크가 성공적으로 벡터 DB에 저장되었습니다."
    }
    
except Exception as e:
    # 상세한 에러 트레이스백을 콘솔과 응답에 모두 전달
    error_details = traceback.format_exc()
    print(f"[Upload API 에러] {error_details}")
    return JSONResponse(status_code=500, content={"error": str(e), "details": error_details})


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
"""질의 확장 -> 벡터 검색 -> LLM 스트리밍 답변 생성 파이프라인"""
try:
# ====================================================
# 1. 질의 확장 (Query Augmentation)
# ====================================================
# 요구사항에 명시된 질의 확장 시스템 메시지 설정
augmentation_system_msg = (
"기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
"대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."
)

    # 프롬프트 템플릿 설계 (과거 대화 내역 동적 바인딩)
    augmentation_prompt = ChatPromptTemplate.from_messages([
        ("system", augmentation_system_msg),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    
    # 체인 연결
    query_augmentation_chain = augmentation_prompt | llm_fast | StrOutputParser()
    
    # 클라이언트에서 전달받은 history 변환
    formatted_history = []
    for h in request.history:
        role = "human" if h["role"] == "user" else "ai"
        formatted_history.append((role, h["content"]))
        
    # 체인 실행 및 증강된 질의 획득
    try:
        augmented_query = query_augmentation_chain.invoke({
            "chat_history": formatted_history,
            "query": request.query
        })
    except Exception as e:
        raise Exception(f"질의 확장 과정에서 LLM 호출 오류가 발생했습니다: {str(e)}")
    
    # ====================================================
    # 2. 벡터 DB 검색 (Supabase 리트리버)
    # ====================================================
    try:
        query_embedding = embeddings.embed_query(augmented_query)
        search_res = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_count": 3}).execute()
    except Exception as e:
        raise Exception(f"Supabase 벡터 검색 중 오류가 발생했습니다: {str(e)}")
    
    contexts = []
    sources = []
    if search_res.data:
        for row in search_res.data:
            contexts.append(row["content"])
            source_info = f"{row['metadata'].get('source_file', '알 수 없는 출처')} (p.{row['metadata'].get('page', '?')})"
            if source_info not in sources:
                sources.append(source_info)
            
    context_text = "\n\n---\n\n".join(contexts)

    # ====================================================
    # 3. 답변 생성 및 스트리밍 전송
    # ====================================================
    # 요구사항에 명시된 페르소나 적용
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n\n[참고 문서]\n{context}"),
        ("human", "{query}")
    ])
    
    generation_chain = answer_prompt | llm_streaming | StrOutputParser()

    async def generate_stream():
        try:
            # 1단계: RAG 검색 컨텍스트 및 출처 정보를 클라이언트에 먼저 전송
            init_data = {
                "type": "context", 
                "augmented_query": augmented_query,
                "sources": sources,
                "contexts": contexts
            }
            yield f"data: {json.dumps(init_data)}\n\n"
            
            # 2단계: StreamHandler 역할을 하는 astream을 통해 실시간 청크 전송
            async for chunk in generation_chain.astream({"context": context_text, "query": augmented_query}):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            # 스트리밍 도중 발생한 에러를 클라이언트로 전송
            error_msg = f"답변 생성 및 스트리밍 중 오류 발생: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

except Exception as e:
    error_details = traceback.format_exc()
    print(f"[Chat API 에러] {error_details}")
    return JSONResponse(status_code=500, content={"error": str(e), "details": error_details})
