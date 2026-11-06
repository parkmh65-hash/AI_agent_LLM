"""
================================================================================
[FastAPI 백엔드] FAISS Index + Multi-Query + Hybrid Search + RAG Fusion 서버
파일 경로: advanced_rag/server/main.py
설명:
  1. FAISS Index 구축 계층: IndexFlatL2(전수 L2 거리 계산) 및 IndexHNSWFlat(그래프 기반 근사 탐색) 지원
  2. 다중 쿼리 확장 계층 (Multi-Query Layer): 단일 사용자 질의를 다각도 파생 쿼리로 확장
  3. 하이브리드 검색 계층 (Hybrid Search Layer): 확장 쿼리별 FAISS(Dense) + BM25(Sparse) 병행 검색
  4. RRF 순위 융합 계층 (RAG Fusion & RRF Layer): Reciprocal Rank Fusion 공식으로 상위 문서 재정렬
  5. 실시간 스트리밍 응답 (NDJSON Stream): GAS(Code.gs)의 StreamHandler와 연동하여 단계별 진행 상황 전송
================================================================================
"""

import base64
import hashlib
import io
import json
import os
import re
import time
import traceback
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# Google Gemini SDK 로드 (환경변수 GEMINI_API_KEY 또는 GOOGLE_API_KEY 사용)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ==============================================================================
# 1. 알고리즘 복잡도 메타데이터 정의 (UI 및 상태 모니터링 표시용)
# ==============================================================================
ALGORITHM_COMPLEXITY: Dict[str, Dict[str, str]] = {
    "faiss_flat_l2": {
        "name": "FAISS IndexFlatL2 (Exact Dense Search)",
        "time_complexity": "O(N · d)",
        "space_complexity": "O(N · d)",
        "description": "모든 문서 벡터(N)에 대해 d차원 유클리드 거리(L2)를 전수 계산하여 100% 정확한 최근접 이웃을 탐색합니다."
    },
    "faiss_hnsw_flat": {
        "name": "FAISS IndexHNSWFlat (Graph ANN Search)",
        "time_complexity": "O(log N · d)",
        "space_complexity": "O(N · (d + M))",
        "description": "다계층 소세계 그래프(Hierarchical Navigable Small World)를 통해 로그 시간 복잡도로 초고속 근사 최근접 탐색을 수행합니다."
    },
    "bm25_sparse": {
        "name": "BM25Okapi (Sparse Keyword Search)",
        "time_complexity": "O(|q| · N_doc)",
        "space_complexity": "O(V + N_tokens)",
        "description": "TF-IDF와 문서 길이 정규화를 결합하여 쿼리 토큰(|q|)과 정확히 일치하는 키워드 매칭 점수를 산출합니다."
    },
    "multi_query": {
        "name": "Multi-Query Expansion (LLM Perspective Generation)",
        "time_complexity": "O(M_q · T_LLM)",
        "space_complexity": "O(M_q)",
        "description": "단일 질의를 M_q개의 서로 다른 관점 쿼리로 확장하여 어휘 불일치(Lexical Chasm) 및 검색 누락을 방지합니다."
    },
    "rrf_fusion": {
        "name": "RAG Fusion & Reciprocal Rank Fusion (RRF)",
        "time_complexity": "O(M_q · K log K)",
        "space_complexity": "O(U_cand)",
        "description": "RRF 공식 [Score(d) = Σ 1 / (k + rank(d))]을 통해 서로 다른 스케일의 FAISS 거리와 BM25 점수를 순위 기반으로 통합·재정렬합니다."
    }
}

# ==============================================================================
# 2. 데이터 모델 (Pydantic Schemas)
# ==============================================================================
class BuildDBRequest(BaseModel):
    """PDF 파일 기반 벡터 DB(FAISS + BM25) 구축 요청 모델"""
    filename: str = Field(..., description="업로드된 PDF 파일명")
    pdf_base64: Optional[str] = Field(None, description="Base64로 인코딩된 PDF 바이너리 데이터")
    raw_text: Optional[str] = Field(None, description="PDF 대신 직접 전달할 텍스트(테스트/폴백용)")
    index_type: Literal["IndexFlatL2", "IndexHNSWFlat"] = Field(
        "IndexFlatL2", description="FAISS 인덱스 알고리즘 선택 (IndexFlatL2 또는 IndexHNSWFlat)"
    )
    chunk_size: int = Field(500, ge=100, le=2000, description="청크 분할 글자 수")
    chunk_overlap: int = Field(100, ge=0, le=500, description="청크 간 중첩 글자 수")
    stream: bool = Field(True, description="GAS StreamHandler 연동용 NDJSON 스트리밍 여부")


class QueryRequest(BaseModel):
    """Multi-Query + Hybrid Search + RAG Fusion 질의 요청 모델"""
    query: str = Field(..., min_length=1, description="사용자의 입력 질의")
    top_k: int = Field(4, ge=1, le=10, description="최종 반환할 상위 RRF 융합 문서 수")
    num_queries: int = Field(3, ge=2, le=5, description="다중 쿼리 확장 개수")
    rrf_k: int = Field(60, ge=1, le=200, description="RRF 공식의 상수 k (기본값 60)")
    dense_weight: float = Field(1.0, ge=0.0, le=5.0, description="FAISS Dense 검색 가중치")
    sparse_weight: float = Field(1.0, ge=0.0, le=5.0, description="BM25 Sparse 검색 가중치")
    stream: bool = Field(True, description="GAS StreamHandler 연동용 NDJSON 스트리밍 여부")


# ==============================================================================
# 3. 텍스트 토크나이저 및 임베딩 유틸리티
# ==============================================================================
def tokenize_ko_en(text: str) -> List[str]:
    """
    한국어 및 영어 하이브리드 토크나이저 (BM25 키워드 검색용)
    - 어절 단위 토큰과 한국어 2-gram(바이그램)을 함께 추출하여 형태소 분석기 없이도
      한국어 조사 결합에 강건한 Sparse 키워드 매칭을 수행합니다.
    - 시간 복잡도: O(L) (L은 텍스트 길이)
    """
    cleaned = re.sub(r"[^\w\s가-힣]", " ", text.lower())
    words = [w for w in cleaned.split() if len(w) >= 1]
    tokens: List[str] = list(words)
    # 한국어 복합명사/조사 대응을 위한 2-gram 서브워드 추가
    for word in words:
        if len(word) >= 2 and re.search(r"[가-힣]", word):
            for i in range(len(word) - 1):
                tokens.append(word[i : i + 2])
    return tokens


def get_gemini_client() -> Optional[Any]:
    """환경변수에서 API 키를 읽어 Gemini 클라이언트를 반환합니다."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if GENAI_AVAILABLE and api_key:
        return genai.Client(api_key=api_key)
    return None


def compute_fallback_embedding(text: str, dim: int = 768) -> np.ndarray:
    """
    Gemini API 키가 없거나 네트워크/할당량 오류 발생 시 동작하는 결정론적(Deterministic) 임베딩 생성기.
    토큰 및 문자 n-gram 해싱 투영(Feature Hashing / Random Projection)을 통해
    의미적·어휘적 유사도가 보존되는 정규화 밀집 벡터를 생성합니다.
    - 시간 복잡도: O(L + d)
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = tokenize_ko_en(text)
    if not tokens:
        tokens = [text]
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx1 = h % dim
        idx2 = (h >> 16) % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx1] += sign * 1.0
        vec[idx2] += sign * 0.5
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_texts(texts: List[str], dim: int = 768) -> tuple[np.ndarray, str]:
    """
    문서 또는 쿼리 리스트를 고차원 밀집 벡터(Dense Vector) 행렬로 변환합니다.
    1순위: Google Gemini Embedding 모델 (`text-embedding-004`)
    2순위: Feature Hashing 기반 로컬 결정론적 임베딩 (폴백)
    """
    client = get_gemini_client()
    if client is not None:
        try:
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=texts,
            )
            vectors = [
                np.array(emb.values, dtype=np.float32)
                for emb in response.embeddings
            ]
            matrix = np.vstack(vectors).astype(np.float32)
            # L2 정규화 수행 (유클리드 거리와 코사인 유사도 일관성 유지)
            faiss.normalize_L2(matrix)
            return matrix, "gemini:text-embedding-004"
        except Exception as e:
            print(f"[WARN] Gemini 임베딩 호출 실패, 로컬 해시 임베딩으로 전환합니다: {e}")

    # 폴백 임베딩 수행
    matrix = np.vstack([compute_fallback_embedding(t, dim=dim) for t in texts]).astype(np.float32)
    return matrix, "local:feature-hashing-768d"


# ==============================================================================
# 4. 전역 인메모리 하이브리드 저장소 (FAISS + BM25 + 문서 메타데이터)
# ==============================================================================
class HybridVectorStore:
    """
    FAISS 인덱스(IndexFlatL2 / IndexHNSWFlat)와 BM25 인덱스를 통합 관리하는 클래스
    """
    def __init__(self) -> None:
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.index_type: str = "IndexFlatL2"
        self.dimension: int = 768
        self.embedding_backend: str = "uninitialized"
        self.source_filename: str = ""
        self.last_build_time_ms: float = 0.0
        self.hnsw_m: int = 32  # HNSW 그래프 이웃 연결 수

    def is_ready(self) -> bool:
        """인덱스가 정상 구축되어 검색 가능한 상태인지 확인합니다."""
        return (
            self.faiss_index is not None
            and self.bm25_index is not None
            and len(self.documents) > 0
        )

    def build_indices(
        self,
        chunks: List[Dict[str, Any]],
        vectors: np.ndarray,
        index_type: Literal["IndexFlatL2", "IndexHNSWFlat"],
        embedding_backend: str,
        filename: str,
    ) -> Dict[str, Any]:
        """
        추출된 문서 청크와 임베딩 벡터로 FAISS 인덱스 및 BM25 인덱스를 구축합니다.
        """
        t0 = time.perf_counter()
        num_docs, dim = vectors.shape
        self.dimension = dim
        self.index_type = index_type
        self.embedding_backend = embedding_backend
        self.source_filename = filename
        self.documents = chunks

        # 1. FAISS 인덱스 생성 (IndexFlatL2 또는 IndexHNSWFlat)
        if index_type == "IndexHNSWFlat":
            # HNSW 그래프 기반 근사 최근접 이웃 인덱스: 탐색 시간 복잡도 O(log N * d)
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32
        else:
            # 전수 L2 거리 계산 인덱스: 탐색 시간 복잡도 O(N * d)
            index = faiss.IndexFlatL2(dim)

        # FAISS 인덱스에 벡터 적재
        index.add(vectors)
        self.faiss_index = index

        # 2. Sparse 키워드 검색을 위한 BM25 인덱스 구축
        self.tokenized_corpus = [tokenize_ko_en(doc["text"]) for doc in chunks]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        self.last_build_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "num_chunks": num_docs,
            "dimension": dim,
            "index_type": index_type,
            "embedding_backend": embedding_backend,
            "build_time_ms": self.last_build_time_ms,
        }


# 싱글톤 벡터 스토어 인스턴스
VECTOR_STORE = HybridVectorStore()


# ==============================================================================
# 5. 핵심 파이프라인 함수 (PDF 파싱 / Multi-Query / Hybrid Search / RRF / 답변 생성)
# ==============================================================================
def extract_and_chunk_pdf(
    pdf_bytes: Optional[bytes],
    raw_text: Optional[str],
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    PDF 바이너리(또는 텍스트)에서 페이지별로 텍스트를 추출하고 슬라이딩 윈도우 방식으로 청크를 분할합니다.
    출처 표기를 위해 각 청크에 파일명(source), 페이지 번호(page), 청크 번호(chunk_id) 메타데이터를 기록합니다.
    """
    pages_data: List[Dict[str, Any]] = []

    if pdf_bytes:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for idx, page in enumerate(reader.pages):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages_data.append({"page": idx + 1, "text": page_text})
        except Exception as e:
            raise ValueError(f"PDF 파일 파싱 중 오류가 발생했습니다: {str(e)}")
    elif raw_text and raw_text.strip():
        pages_data.append({"page": 1, "text": raw_text.strip()})

    if not pages_data:
        raise ValueError(
            "PDF 문서에서 추출된 텍스트가 없습니다. 스캔본(이미지) PDF이거나 빈 문서인지 확인해 주세요."
        )

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1
    step = max(1, chunk_size - chunk_overlap)

    for page_item in pages_data:
        page_num = page_item["page"]
        text = re.sub(r"\s+", " ", page_item["text"]).strip()
        for start in range(0, len(text), step):
            snippet = text[start : start + chunk_size].strip()
            if len(snippet) < 20 and len(chunks) > 0:
                continue
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_counter:03d}",
                    "text": snippet,
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "chunk_id": f"chunk_{chunk_counter:03d}",
                        "char_length": len(snippet),
                        "start_offset": start,
                    },
                }
            )
            chunk_counter += 1
            if start + chunk_size >= len(text):
                break

    return chunks


def generate_multi_queries(original_query: str, num_queries: int = 3) -> tuple[List[str], str]:
    """
    [다중 쿼리 확장 계층 - Multi-Query Layer]
    사용자의 단일 입력 쿼리를 LLM을 통해 서로 다른 관점(개념적 정의, 핵심 키워드, 실무/적용 관점 등)의
    파생 쿼리로 확장하여 단일 쿼리 검색의 어휘 불일치 한계와 검색 누락을 방지합니다.
    """
    client = get_gemini_client()
    prompt = (
        f"당신은 RAG 검색 최적화 전문가입니다.\n"
        f"사용자의 원본 질의에 대해 문서 검색 재현율(Recall)을 극대화할 수 있도록, "
        f"서로 다른 관점(1: 핵심 개념/정의 관점, 2: 세부 기술/키워드 관점, 3: 문제 해결/적용 사례 관점)의 "
        f"파생 검색 쿼리 {num_queries}개를 생성하세요.\n"
        f"원본 질의: \"{original_query}\"\n"
        f"출력 형식: 번호나 불릿 없이 한 줄에 하나씩 파생 쿼리만 출력하세요."
    )

    if client is not None:
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            lines = [
                re.sub(r"^[\d\-\*\.\)\s]+", "", line).strip()
                for line in (resp.text or "").splitlines()
                if line.strip()
            ]
            derived = [q for q in lines if q and q != original_query][:num_queries]
            if derived:
                return [original_query] + derived, "gemini-2.5-flash"
        except Exception as e:
            print(f"[WARN] Gemini Multi-Query 생성 실패, 규칙 기반 다각도 쿼리 확장으로 전환: {e}")

    # 폴백: 규칙 기반 다각도 쿼리 확장
    core_terms = " ".join(tokenize_ko_en(original_query)[:5])
    fallback_queries = [
        original_query,
        f"{original_query} 핵심 개념 및 정의",
        f"{original_query} 주요 특징 및 작동 원리",
        f"{core_terms} 관련 상세 내용 및 사례",
    ]
    return fallback_queries[: num_queries + 1], "rule-based-expansion"


def run_hybrid_search_and_rrf(
    expanded_queries: List[str],
    top_k: int = 4,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    [하이브리드 검색 계층 (FAISS + BM25) & RRF 순위 융합 계층 (RAG Fusion)]
    1. 확장된 각 쿼리마다 FAISS(Dense Vector) 검색과 BM25(Sparse Keyword) 검색을 병행 수행합니다.
    2. 수집된 다중 경로 검색 결과들의 순위(Rank)를 Reciprocal Rank Fusion(RRF) 공식으로 통합합니다:
       RRF_Score(d) = Σ_{q ∈ Queries} [ w_dense / (k + rank_FAISS(q, d)) + w_sparse / (k + rank_BM25(q, d)) ]
    3. 최종 RRF 점수 기준으로 상위 top_k 문서를 재정렬(Re-ranking)하여 반환합니다.
    """
    if not VECTOR_STORE.is_ready():
        raise ValueError("벡터 DB가 아직 생성되지 않았습니다. 먼저 PDF 파일을 선택하고 벡터 DB를 생성해 주세요.")

    num_docs = len(VECTOR_STORE.documents)
    candidate_k = min(num_docs, max(top_k * 3, 10))

    # 모든 확장 쿼리를 한 번에 임베딩
    query_vectors, _ = embed_texts(expanded_queries, dim=VECTOR_STORE.dimension)

    # 문서별 RRF 누적 점수 및 상세 검색 이력 저장소
    rrf_scores: Dict[int, float] = {}
    doc_provenance: Dict[int, Dict[str, Any]] = {}
    per_query_logs: List[Dict[str, Any]] = []

    for q_idx, query_text in enumerate(expanded_queries):
        q_vec = query_vectors[q_idx : q_idx + 1]

        # ----------------------------------------------------------------------
        # (A) FAISS Dense Vector 검색: 유클리드 L2 거리 기반 의미적 유사도 탐색
        # ----------------------------------------------------------------------
        distances, indices = VECTOR_STORE.faiss_index.search(q_vec, candidate_k)
        dense_hits: List[Dict[str, Any]] = []

        for rank_idx, (doc_id, l2_dist) in enumerate(zip(indices[0], distances[0])):
            if doc_id < 0 or doc_id >= num_docs:
                continue
            rank = rank_idx + 1
            # L2 거리를 0~1 범위의 유사도 점수로 변환: sim = 1 / (1 + L2_dist)
            dense_sim = float(1.0 / (1.0 + max(0.0, float(l2_dist))))
            rrf_contrib = dense_weight * (1.0 / (rrf_k + rank))

            rrf_scores[int(doc_id)] = rrf_scores.get(int(doc_id), 0.0) + rrf_contrib
            prov = doc_provenance.setdefault(
                int(doc_id),
                {
                    "best_faiss_rank": rank,
                    "best_faiss_similarity": round(dense_sim, 4),
                    "min_l2_distance": round(float(l2_dist), 4),
                    "best_bm25_rank": None,
                    "max_bm25_score": 0.0,
                    "matched_queries": set(),
                    "rrf_breakdown": [],
                },
            )
            if rank < prov["best_faiss_rank"]:
                prov["best_faiss_rank"] = rank
                prov["best_faiss_similarity"] = round(dense_sim, 4)
                prov["min_l2_distance"] = round(float(l2_dist), 4)
            prov["matched_queries"].add(query_text)
            prov["rrf_breakdown"].append(
                f"Q{q_idx+1}-FAISS(순위:{rank}, +{_fmt(rrf_contrib)})"
            )
            dense_hits.append({"chunk_id": VECTOR_STORE.documents[int(doc_id)]["chunk_id"], "rank": rank})

        # ----------------------------------------------------------------------
        # (B) BM25 Sparse Keyword 검색: 정확한 어휘 매칭 점수 산출
        # ----------------------------------------------------------------------
        q_tokens = tokenize_ko_en(query_text)
        bm25_raw_scores = VECTOR_STORE.bm25_index.get_scores(q_tokens)
        ranked_bm25_indices = np.argsort(bm25_raw_scores)[::-1][:candidate_k]
        sparse_hits: List[Dict[str, Any]] = []

        for rank_idx, doc_id in enumerate(ranked_bm25_indices):
            bm25_score = float(bm25_raw_scores[doc_id])
            rank = rank_idx + 1
            rrf_contrib = sparse_weight * (1.0 / (rrf_k + rank))

            rrf_scores[int(doc_id)] = rrf_scores.get(int(doc_id), 0.0) + rrf_contrib
            prov = doc_provenance.setdefault(
                int(doc_id),
                {
                    "best_faiss_rank": None,
                    "best_faiss_similarity": 0.0,
                    "min_l2_distance": None,
                    "best_bm25_rank": rank,
                    "max_bm25_score": round(bm25_score, 4),
                    "matched_queries": set(),
                    "rrf_breakdown": [],
                },
            )
            if prov["best_bm25_rank"] is None or rank < prov["best_bm25_rank"]:
                prov["best_bm25_rank"] = rank
            if bm25_score > prov["max_bm25_score"]:
                prov["max_bm25_score"] = round(bm25_score, 4)
            prov["matched_queries"].add(query_text)
            prov["rrf_breakdown"].append(
                f"Q{q_idx+1}-BM25(순위:{rank}, +{_fmt(rrf_contrib)})"
            )
            sparse_hits.append({"chunk_id": VECTOR_STORE.documents[int(doc_id)]["chunk_id"], "rank": rank})

        per_query_logs.append(
            {
                "query_index": q_idx + 1,
                "query": query_text,
                "faiss_top3": dense_hits[:3],
                "bm25_top3": sparse_hits[:3],
            }
        )

    # --------------------------------------------------------------------------
    # (C) RRF 점수 기준 내림차순 정렬 (Re-ranking)
    # --------------------------------------------------------------------------
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

    reranked_documents: List[Dict[str, Any]] = []
    for final_rank, doc_id in enumerate(sorted_doc_ids, start=1):
        doc = VECTOR_STORE.documents[doc_id]
        prov = doc_provenance[doc_id]
        reranked_documents.append(
            {
                "final_rank": final_rank,
                "chunk_id": doc["chunk_id"],
                "snippet": doc["text"],
                "metadata": doc["metadata"],
                "scores": {
                    "rrf_score": round(rrf_scores[doc_id], 6),
                    "best_faiss_rank": prov["best_faiss_rank"],
                    "faiss_similarity": prov["best_faiss_similarity"],
                    "l2_distance": prov["min_l2_distance"],
                    "best_bm25_rank": prov["best_bm25_rank"],
                    "bm25_score": prov["max_bm25_score"],
                },
                "matched_queries_count": len(prov["matched_queries"]),
                "rrf_Contributions": prov["rrf_breakdown"][:6],
            }
        )

    return {
        "reranked_documents": reranked_documents,
        "per_query_logs": per_query_logs,
        "total_candidates_evaluated": len(rrf_scores),
    }


def _fmt(val: float) -> str:
    return f"{val:.4f}"


def synthesize_llm_answer(query: str, reranked_docs: List[Dict[str, Any]]) -> tuple[str, str]:
    """
    RRF로 재정렬된 최상위 문맥(Context)만을 근거로 LLM 최종 답변을 생성하며,
    각 문단이나 핵심 근거마다 [출처: 파일명 p.페이지 (청크ID)] 형태로 명확히 출처를 표기합니다.
    """
    if not reranked_docs:
        return "검색된 문서가 없어 답변을 생성할 수 없습니다.", "none"

    context_blocks = []
    for doc in reranked_docs:
        meta = doc["metadata"]
        source_tag = f"[출처: {meta['source']} p.{meta['page']} ({doc['chunk_id']}, RRF={doc['scores']['rrf_score']:.4f})]"
        context_blocks.append(f"{source_tag}\n{doc['snippet']}")

    joined_context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "당신은 신뢰할 수 있는 기업용 RAG AI 어시스턴트입니다.\n"
        "아래 제공된 [검색된 핵심 문서(RRF 상위 결과)]만을 근거로 사용자의 질문에 체계적이고 정확하게 한국어로 답변하세요.\n"
        "반드시 답변 본문 내에 사용한 근거의 출처를 `[출처: 파일명 p.페이지 (chunk_id)]` 형식으로 명시하세요.\n\n"
        f"### [검색된 핵심 문서(RRF 상위 결과)]\n{joined_context}\n\n"
        f"### [사용자 질문]\n{query}\n\n"
        "### [작성 지침]\n"
        "1. 핵심 요약을 먼저 제시하고 세부 내용을 불릿 포인트로 정리하세요.\n"
        "2. 각 핵심 설명 뒤에 반드시 `[출처: ...]`를 표기하세요.\n"
        "3. 문서에 없는 내용은 추측하지 말고 문서에서 확인된 사실만 기술하세요."
    )

    client = get_gemini_client()
    if client is not None:
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if resp.text:
                return resp.text.strip(), "gemini-2.5-flash"
        except Exception as e:
            print(f"[WARN] Gemini 답변 생성 호출 실패, 검색 문서 기반 요약 폴백 실행: {e}")

    # Gemini API 키 미설정 시: RRF 상위 문서들을 구조화하여 출처와 함께 요약 응답 생성
    lines = [
        f"### 질문에 대한 RAG Fusion 기반 핵심 답변 요약",
        f"질의하신 **\"{query}\"**에 대하여 RRF(Reciprocal Rank Fusion) 순위 융합 알고리즘으로 선별된 상위 {len(reranked_docs)}개 핵심 문맥 기반 답변입니다.\n",
    ]
    for doc in reranked_docs:
        meta = doc["metadata"]
        sc = doc["scores"]
        lines.append(
            f"- **핵심 근거 #{doc['final_rank']}** (RRF 점수: `{sc['rrf_score']:.4f}`, "
            f"FAISS 최고순위: `{sc['best_faiss_rank']}위`, BM25 최고순위: `{sc['best_bm25_rank']}위`)\n"
            f"  > {doc['snippet']}\n"
            f"  **[출처: {meta['source']} p.{meta['page']} ({doc['chunk_id']})]**"
        )
    lines.append(
        "\n*참고: 서버에 `GEMINI_API_KEY` 환경변수를 설정하면 Gemini 2.5 Flash 생성형 요약 답변이 활성화됩니다.*"
    )
    return "\n\n".join(lines), "extractive-rag-fallback"


# ==============================================================================
# 6. FastAPI 애플리케이션 및 엔드포인트 정의
# ==============================================================================
app = FastAPI(
    title="Advanced RAG Server (FAISS + Multi-Query + Hybrid Search + RRF)",
    description="FAISS Dense Vector + BM25 Sparse Search + Multi-Query + Reciprocal Rank Fusion API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """모든 예기치 못한 서버 예외를 표준화된 상세 에러 구조로 반환합니다."""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "hint": "요청 데이터 형식(PDF 인코딩 또는 쿼리 파라미터) 및 서버 환경변수를 확인하세요.",
            "path": str(request.url.path),
            "traceback": traceback.format_exc().splitlines()[-3:],
        },
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    서버 동작 상태, 현재 적재된 FAISS/BM25 인덱스 현황 및 알고리즘 복잡도 정보를 반환합니다.
    """
    return {
        "status": "ok",
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gemini_api_configured": get_gemini_client() is not None,
        "vector_db": {
            "is_ready": VECTOR_STORE.is_ready(),
            "source_filename": VECTOR_STORE.source_filename,
            "num_chunks": len(VECTOR_STORE.documents),
            "index_type": VECTOR_STORE.index_type,
            "dimension": VECTOR_STORE.dimension,
            "embedding_backend": VECTOR_STORE.embedding_backend,
            "last_build_time_ms": VECTOR_STORE.last_build_time_ms,
        },
        "algorithm_complexity": ALGORITHM_COMPLEXITY,
    }


@app.post("/build_db")
async def build_vector_db(req: BuildDBRequest):
    """
    PDF 파일을 파싱하여 청크로 분할하고 FAISS 인덱스(IndexFlatL2 / IndexHNSWFlat)와
    BM25 키워드 인덱스를 구축합니다.
    - req.stream == True 인 경우: GAS StreamHandler가 실시간 파싱할 수 있도록 NDJSON 스트림으로 단계별 진행 현황을 전송합니다.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        t_start = time.perf_counter()
        try:
            # Step 1: PDF 디코딩 및 유효성 검사
            yield json.dumps(
                {
                    "type": "progress",
                    "step": 1,
                    "total_steps": 4,
                    "percent": 15,
                    "stage": "PDF_DECODE",
                    "message": f"'{req.filename}' 파일 수신 완료 및 바이너리 디코딩 중...",
                },
                ensure_ascii=False,
            ) + "\n"

            pdf_bytes: Optional[bytes] = None
            if req.pdf_base64:
                clean_b64 = req.pdf_base64.split(",")[-1].strip()
                pdf_bytes = base64.b64decode(clean_b64)

            # Step 2: PDF 텍스트 추출 및 슬라이딩 윈도우 청크 분할
            chunks = extract_and_chunk_pdf(
                pdf_bytes=pdf_bytes,
                raw_text=req.raw_text,
                filename=req.filename,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
            yield json.dumps(
                {
                    "type": "progress",
                    "step": 2,
                    "total_steps": 4,
                    "percent": 45,
                    "stage": "CHUNKING_COMPLETE",
                    "message": f"문서 파싱 및 청크 분할 완료: 총 {len(chunks)}개 청크 생성 (청크 크기={req.chunk_size}, 중첩={req.chunk_overlap})",
                },
                ensure_ascii=False,
            ) + "\n"

            # Step 3: 고차원 임베딩 벡터 생성
            texts = [c["text"] for c in chunks]
            vectors, emb_backend = embed_texts(texts)
            yield json.dumps(
                {
                    "type": "progress",
                    "step": 3,
                    "total_steps": 4,
                    "percent": 75,
                    "stage": "EMBEDDING_COMPLETE",
                    "message": f"고차원 밀집 벡터(Dense Embedding) 생성 완료: {vectors.shape[0]}개 × {vectors.shape[1]}차원 ({emb_backend})",
                },
                ensure_ascii=False,
            ) + "\n"

            # Step 4: FAISS Index & BM25 Sparse Index 동시 구축
            build_stats = VECTOR_STORE.build_indices(
                chunks=chunks,
                vectors=vectors,
                index_type=req.index_type,
                embedding_backend=emb_backend,
                filename=req.filename,
            )
            total_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

            active_faiss_complexity = (
                ALGORITHM_COMPLEXITY["faiss_hnsw_flat"]
                if req.index_type == "IndexHNSWFlat"
                else ALGORITHM_COMPLEXITY["faiss_flat_l2"]
            )

            yield json.dumps(
                {
                    "type": "complete",
                    "step": 4,
                    "total_steps": 4,
                    "percent": 100,
                    "stage": "INDEX_READY",
                    "message": f"FAISS({req.index_type}) 및 BM25 하이브리드 인덱스 구축 완료! (총 {total_elapsed_ms}ms 소요)",
                    "data": {
                        "status": "success",
                        "filename": req.filename,
                        **build_stats,
                        "total_elapsed_ms": total_elapsed_ms,
                        "applied_complexity": {
                            "dense_index": active_faiss_complexity,
                            "sparse_index": ALGORITHM_COMPLEXITY["bm25_sparse"],
                        },
                        "sample_chunks": chunks[:2],
                    },
                },
                ensure_ascii=False,
            ) + "\n"

        except Exception as e:
            yield json.dumps(
                {
                    "type": "error",
                    "stage": "BUILD_DB_ERROR",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "hint": "유효한 텍스트가 포함된 PDF 파일인지 확인하거나 다른 PDF 파일로 다시 시도해 주세요.",
                },
                ensure_ascii=False,
            ) + "\n"

    if req.stream:
        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    # 비스트리밍 요청 시 모든 이벤트를 수집하여 단일 JSON으로 반환
    events = []
    async for line in event_generator():
        events.append(json.loads(line))
    last_event = events[-1]
    if last_event.get("type") == "error":
        raise HTTPException(status_code=400, detail=last_event)
    return {"status": "success", "events": events, "result": last_event.get("data")}


@app.post("/query")
async def query_rag_fusion(req: QueryRequest):
    """
    [핵심 파이프라인 엔드포인트]
    1. Multi-Query Layer: 단일 질의를 다각도 파생 쿼리로 확장
    2. Hybrid Search Layer: 각 확장 쿼리마다 FAISS(Dense) + BM25(Sparse) 병행 검색
    3. RAG Fusion (RRF) Layer: Reciprocal Rank Fusion 공식으로 상위 문서 재정렬
    4. Answer Synthesis: 선별된 상위 문서 스니펫을 기반으로 출처 포함 최종 답변 생성
    """
    async def query_stream_generator() -> AsyncGenerator[str, None]:
        t_start = time.perf_counter()
        try:
            if not VECTOR_STORE.is_ready():
                raise ValueError(
                    "벡터 DB가 비어 있습니다. 먼저 PDF 파일을 선택하고 [벡터 DB 생성] 버튼을 눌러 인덱스를 구축해 주세요."
                )

            # Stage 1: 다중 쿼리 확장 (Multi-Query Layer)
            t_mq = time.perf_counter()
            expanded_queries, mq_backend = generate_multi_queries(
                req.query, num_queries=req.num_queries
            )
            mq_ms = round((time.perf_counter() - t_mq) * 1000, 2)

            yield json.dumps(
                {
                    "type": "progress",
                    "step": 1,
                    "total_steps": 4,
                    "percent": 25,
                    "stage": "MULTI_QUERY_EXPANSION",
                    "message": f"[Multi-Query 계층] 원본 질의를 {len(expanded_queries)}개의 다각도 쿼리로 확장 완료 ({mq_ms}ms)",
                    "expanded_queries": expanded_queries,
                },
                ensure_ascii=False,
            ) + "\n"

            # Stage 2 & 3: 하이브리드 검색(FAISS + BM25) 및 RRF 순위 융합
            t_search = time.perf_counter()
            fusion_result = run_hybrid_search_and_rrf(
                expanded_queries=expanded_queries,
                top_k=req.top_k,
                rrf_k=req.rrf_k,
                dense_weight=req.dense_weight,
                sparse_weight=req.sparse_weight,
            )
            search_ms = round((time.perf_counter() - t_search) * 1000, 2)

            yield json.dumps(
                {
                    "type": "progress",
                    "step": 2,
                    "total_steps": 4,
                    "percent": 60,
                    "stage": "HYBRID_SEARCH_AND_RRF",
                    "message": (
                        f"[Hybrid Search & RRF 계층] {len(expanded_queries)}개 쿼리 × (FAISS {VECTOR_STORE.index_type} + BM25) "
                        f"검색 후 RRF(k={req.rrf_k}) 재정렬 완료 ({search_ms}ms)"
                    ),
                    "total_candidates": fusion_result["total_candidates_evaluated"],
                },
                ensure_ascii=False,
            ) + "\n"

            # Stage 4: LLM 최종 답변 생성 (출처 포함)
            t_llm = time.perf_counter()
            llm_answer, llm_backend = synthesize_llm_answer(
                req.query, fusion_result["reranked_documents"]
            )
            llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)

            active_faiss_key = (
                "faiss_hnsw_flat"
                if VECTOR_STORE.index_type == "IndexHNSWFlat"
                else "faiss_flat_l2"
            )

            yield json.dumps(
                {
                    "type": "complete",
                    "step": 4,
                    "total_steps": 4,
                    "percent": 100,
                    "stage": "ANSWER_COMPLETE",
                    "message": f"RAG Fusion 검색 및 LLM 답변 생성 완료 (총 {total_ms}ms 소요)",
                    "data": {
                        "status": "success",
                        "original_query": req.query,
                        "expanded_queries": expanded_queries,
                        "multi_query_backend": mq_backend,
                        "llm_backend": llm_backend,
                        "index_info": {
                            "source_filename": VECTOR_STORE.source_filename,
                            "index_type": VECTOR_STORE.index_type,
                            "total_Indexed_chunks": len(VECTOR_STORE.documents),
                            "dimension": VECTOR_STORE.dimension,
                            "rrf_k": req.rrf_k,
                        },
                        "timing_ms": {
                            "multi_query_ms": mq_ms,
                            "hybrid_rrf_ms": search_ms,
                            "llm_synthesis_ms": llm_ms,
                            "total_ms": total_ms,
                        },
                        "algorithm_complexity": {
                            "multi_query": ALGORITHM_COMPLEXITY["multi_query"],
                            "dense_search": ALGORITHM_COMPLEXITY[active_faiss_key],
                            "sparse_search": ALGORITHM_COMPLEXITY["bm25_sparse"],
                            "rrf_fusion": ALGORITHM_COMPLEXITY["rrf_fusion"],
                        },
                        "retrieved_documents": fusion_result["reranked_documents"],
                        "per_query_search_logs": fusion_result["per_query_logs"],
                        "llm_answer": llm_answer,
                    },
                },
                ensure_ascii=False,
            ) + "\n"

        except Exception as e:
            yield json.dumps(
                {
                    "type": "error",
                    "stage": "QUERY_PIPELINE_ERROR",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "hint": "벡터 DB가 생성되어 있는지 확인하고 질의를 다시 시도해 주세요.",
                },
                ensure_ascii=False,
            ) + "\n"

    if req.stream:
        return StreamingResponse(
            query_stream_generator(), media_type="application/x-ndjson"
        )

    events = []
    async for line in query_stream_generator():
        events.append(json.loads(line))
    last_event = events[-1]
    if last_event.get("type") == "error":
        raise HTTPException(status_code=400, detail=last_event)
    return last_event.get("data")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

