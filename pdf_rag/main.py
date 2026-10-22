import os
import json
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from supabase import create_client, Client

app = FastAPI()

# 브라우저(GAS 웹앱)에서 직접 호출할 수 있도록 CORS 전면 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 로드 (Cloud Run 환경 변수에 등록 필요)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # 구글 AI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # 오픈 AI API 키 (임베딩용)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY) # 오픈 AI 임베딩 모델 사용

# LLM 초기화
# 답변 생성용 (실시간 스트리밍 핸들러 적용을 위해 streaming=True 설정)
llm_streaming = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, streaming=True)
# 질의 확장용 (스트리밍 비활성화, 빠른 처리 목적)
llm_fast = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = [] # [{"role": "user", "content": "..."}, ...]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """입력한 PDF 파일을 텍스트로 변환하고 청크 단위로 쪼개어 Supabase 벡터 DB에 저장"""
    try:
        # 임시 파일로 PDF 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # PDF 로드 및 텍스트 변환
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        
        # 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        # 오픈 AI 임베딩 모델 사용 및 벡트 DB 적용 (Supabase)
        for chunk in chunks:
            meta = chunk.metadata
            meta["source_file"] = file.filename
            
            vector = embeddings.embed_query(chunk.page_content)
            supabase.table("documents").insert({
                "content": chunk.page_content,
                "metadata": meta,
                "embedding": vector
            }).execute()

        os.remove(tmp_path)
        return {
            "filename": file.filename, 
            "text": full_text, 
            "message": f"총 {len(chunks)}개의 청크로 벡터 DB 생성 완료"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat_stream(request: QueryRequest):
    """질의 확장 -> 벡터 DB 검색 -> 주어진 청크 기반 LLM 답변 스트리밍"""
    
    # ==========================================
    # 질의 확장 구현하기 (Query Augmentation)
    # ==========================================
    
    # 1. 시스템 메시지 및 프롬프트 설계
    # LLM에게 부여할 역할(의도 파악 및 질문 변환)을 정의하는 시스템 메시지 작성
    system_msg = (
        "당신은 AI 어시스턴트입니다. 이전 대화 내용을 참고하여, "
        "사용자의 최신 질문을 검색 엔진이 완벽하게 이해할 수 있는 하나의 독립적이고 명확한 질문으로 재작성하세요. "
        "질문에 대한 답을 하지 말고 오직 재구성된 질문 문장만 반환하세요."
    )
    
    # 2. 대화 기록(Chat History) 전달 공간 마련
    # 프롬프트 템플릿 내부에 과거의 대화 내역이 통째로 들어갈 수 있는 동적 빈칸(MessagesPlaceholder)을 마련함
    augmentation_prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    
    # 3. Query Augmentation Chain 구성
    # 프롬프트 템플릿, LLM 객체(llm_fast), 출력 파서(StrOutputParser)를 파이프라인처럼 연결함
    query_augmentation_chain = augmentation_prompt | llm_fast | StrOutputParser()
    
    # 요청받은 history 배열을 LangChain 대화 기록 형식으로 변환
    formatted_history = []
    for h in request.history:
        role = "human" if h["role"] == "user" else "ai"
        formatted_history.append((role, h["content"]))
        
    # 4. 체인 실행 (데이터 주입)
    # 1) 현재 사용자의 질문과 2) 이전까지의 대화 기록을 딕셔너리 형태로 묶어서 주입함
    augmented_query = query_augmentation_chain.invoke({
        "chat_history": formatted_history,
        "query": request.query
    })
    
    # 5. 증강된 질의로 벡터 DB 검색 (Supabase 리트리버 개발 적용)
    # 완성된 augmented_query를 리트리버(Supabase RPC 함수)에 전달하여 문서 검색 수행
    query_embedding = embeddings.embed_query(augmented_query)
    search_res = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_count": 3}).execute()
    
    contexts = []
    sources = []
    if search_res.data:
        for row in search_res.data:
            contexts.append(row["content"])
            # 출처 메타데이터 수집
            source_info = f"{row['metadata'].get('source_file', '알 수 없음')} (p.{row['metadata'].get('page', '?')})"
            if source_info not in sources:
                sources.append(source_info)
            
    context_text = "\n\n---\n\n".join(contexts)

    # ==========================================
    # 주어진 청크 기반 LLM 답변 생성 및 실시간 스트리밍
    # ==========================================
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 제공된 [참고 문서]를 바탕으로 질문에 답변하는 전문가입니다.\n\n[참고 문서]\n{context}"),
        ("human", "질문: {query}")
    ])
    
    # StreamHandler 적용을 위해 streaming=True가 적용된 llm_streaming 사용
    generation_chain = answer_prompt | llm_streaming | StrOutputParser()

    async def generate_stream():
        # 클라이언트 분리 표시를 위해 RAG 검색 내용(Contexts)과 출처(Sources)를 가장 먼저 전송
        init_data = {
            "type": "context", 
            "augmented_query": augmented_query,
            "sources": sources,
            "contexts": contexts
        }
        yield f"data: {json.dumps(init_data)}\n\n"
        
        # LLM의 답변 조각(Chunk)을 실시간으로 낚아채어 클라이언트로 전송
        async for chunk in generation_chain.astream({"context": context_text, "query": augmented_query}):
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
        
        yield "data: [DONE]\n\n"

    # Server-Sent Events (SSE) 포맷으로 스트리밍 반환
    return StreamingResponse(generate_stream(), media_type="text/event-stream")
