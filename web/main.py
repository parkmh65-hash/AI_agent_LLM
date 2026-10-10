from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import pytz

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from youtube_search import YoutubeSearch
from langchain_community.document_loaders import YoutubeLoader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 도구 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """지정된 타임존과 위치의 현재 시각을 반환합니다."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timezone} ({location}) 현재 시각: {now}"
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"

@tool
def get_web_search(query: str, search_period: str = "w") -> str:
    """DuckDuckGo를 통해 웹 검색을 수행합니다. search_period는 w(1주일), m(1개월), y(1년)입니다."""
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)
        search = DuckDuckGoSearchResults(
            api_wrapper=wrapper,
            results_separator=";\n"
        )
        return search.invoke(query)
    except Exception as e:
        return f"검색 오류: {str(e)}"

@tool
def get_youtube_search(query: str) -> str:      
    """유튜브 검색을 한 뒤, 영상들의 주요 내용을 문자열로 반환하는 함수."""
    try:
        videos = YoutubeSearch(query, max_results=3).to_dict()
        # 1시간 미만의 영상만 필터링 (mm:ss 형태 필터링 단순화)
        videos = [v for v in videos if len(v['duration'].split(':')) < 3]

        result_text = []
        for v in videos:
            v_url = 'https://youtube.com' + v['url_suffix']
            try:
                loader = YoutubeLoader.from_youtube_url(v_url, language=['ko', 'en'])
                docs = loader.load()
                content = docs[0].page_content[:1500] if docs else "자막 없음"
            except:
                content = "자막 추출 실패"
            
            result_text.append(f"제목: {v['title']}\n링크: {v_url}\n내용 요약: {content}\n")
        
        return "\n".join(result_text) if result_text else "관련 영상을 찾을 수 없습니다."
    except Exception as e:
        return f"유튜브 검색 오류: {str(e)}"


tools = [get_current_time, get_web_search, get_youtube_search]
tool_map = {tool.name: tool for tool in tools}

# 2. 요청 모델 정의
class ChatMessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessageItem] = []

@app.post("/api/chat")
def handle_chat(req: ChatRequest):
    # Gemini 모델 초기화 및 도구 바인딩
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)

    # 메시지 리스트 복원
    messages = [
        SystemMessage(content="너는 시간 조회, 웹 검색, 유튜브 검색 도구를 활용하여 질문에 답하는 어시스턴트이다.")
    ]

    for item in req.history:
        if item.role == "user":
            messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            messages.append(AIMessage(content=item.content))

    messages.append(HumanMessage(content=req.message))

    # 1차 추론 (도구 호출 판단)
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # 도구 호출이 발생했을 경우
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if tool_name in tool_map:
                tool_output = tool_map[tool_name].invoke(tool_args)
            else:
                tool_output = f"도구 '{tool_name}'를 찾을 수 없습니다."
            
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))
        
        # 2차 추론 (도구 실행 결과를 바탕으로 최종 답변)
        final_response = llm_with_tools.invoke(messages)
        reply_text = final_response.content
    else:
        reply_text = ai_msg.content

    return {"reply": reply_text}
