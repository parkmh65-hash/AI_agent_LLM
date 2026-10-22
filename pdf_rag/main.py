import os
import json
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

# 텍스트 추출 및 분할, 임베딩을 위한 LangChain 라이브러리
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from supabase import create_client, Client

app = FastAPI()

# 클라이언트(GAS 등)와의 통신을 위한 CORS 전면 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 로드 (Cloud Run 환경변수에 등록해야 함)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # 구글 LLM용
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # OpenAI 임베딩용
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# API 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# LLM 초기화
# 질의 확장용 모델 (빠른 처리 목적)
llm_fast = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY)
# 답변 생성용 모델 (스트리밍 활성화)
llm_streaming = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY, streaming=True)

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = [] # [{"role": "user", "content": "..."}, ...]

@app.get("/health")
async def health_check():
    """클라이언트(프론트엔드)에서 서버 연동 상태를 확인하기 위한 상태 체크 API"""
    return {"status": "ok", "message": "Server is running."}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """입력한 PDF 파일을 텍스트로 변환하고 청크 단위로 쪼개어 벡터 DB에 저장"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # PDF 로드 및 텍스트 추출
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        
        # 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        # 오픈 AI 임베딩 모델을 사용하여 텍스트를 벡터로 변환 후 Supabase(벡터 DB)에 저장
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
            "message": f"총 {len(chunks)}개의 텍스트 청크가 벡터 DB에 저장되었습니다."
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """질의 확장 및 벡터 검색 후 도시 정책 전문가로서 LLM 답변 스트리밍"""
    
    # ==========================================
    # 1. 질의 확장 (Query Augmentation)
    # ==========================================
    
    # 1. 시스템 메시지 및 프롬프트 설계
    # 요구사항에 명시된 질의 확장용 시스템 메시지 적용
    system_msg = (
        "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
        "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."
    )
    
    # 2. 대화 기록(Chat History) 전달 공간 마련
    augmentation_prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    
    # 3. Query Augmentation Chain 구성 (프롬프트 -> LLM -> 파서)
    query_augmentation_chain = augmentation_prompt | llm_fast | StrOutputParser()
    
    # 전달받은 history 배열을 LangChain 형식의 튜플 리스트로 변환
    formatted_history = []
    for h in request.history:
        role = "human" if h["role"] == "user" else "ai"
        formatted_history.append((role, h["content"]))
        
    # 4. 체인 실행 (데이터 주입)
    # 1) 현재 사용자의 질문과 2) 이전 대화 기록을 딕셔너리로 묶어서 주입
    augmented_query = query_augmentation_chain.invoke({
        "chat_history": formatted_history,
        "query": request.query
    })
    
    # ==========================================
    # 2. 벡터 DB 검색 (Retrieval)
    # ==========================================
    
    # 5. 증강된 질의로 벡터 DB 검색 (Supabase 리트리버 호출)
    query_embedding = embeddings.embed_query(augmented_query)
    search_res = supabase.rpc("match_documents", {"query_embedding": query_embedding, "match_count": 3}).execute()
    
    contexts = []
    sources = []
    if search_res.data:
        for row in search_res.data:
            contexts.append(row["content"])
            source_info = f"{row['metadata'].get('source_file', 'Unknown')} (p.{row['metadata'].get('page', '?')})"
            if source_info not in sources:
                sources.append(source_info)
            
    context_text = "\n\n".join(contexts)

    # ==========================================
    # 3. 답변 생성 및 실시간 스트리밍
    # ==========================================
    
    # 도시 정책 전문가 페르소나 부여
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n\n[참고 문서]\n{context}"),
        ("human", "{query}")
    ])
    generation_chain = answer_prompt | llm_streaming | StrOutputParser()

    async def generate_stream():
        # 첫 번째 스트림: 클라이언트에 RAG 검색 결과 및 출처 정보 전달
        init_data = {
            "type": "context", 
            "augmented_query": augmented_query,
            "sources": sources,
            "contexts": contexts
        }
        yield f"data: {json.dumps(init_data)}\n\n"
        
        # 두 번째 스트림: StreamHandler를 활용하듯 LLM의 응답 청크를 순차적으로 실시간 전송
        async for chunk in generation_chain.astream({"context": context_text, "query": augmented_query}):
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")
