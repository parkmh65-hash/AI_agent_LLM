from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PoemRequest(BaseModel):
    topic: str

@app.post("/api/poem")
def generate_poem(data: PoemRequest):
    # GOOGLE_API_KEY 환경변수 자동 참조
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    prompt = PromptTemplate.from_template("주제 '{topic}'에 대한 짧고 감동적인 시를 한 편 작성해줘.")
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    result = chain.invoke({"topic": data.topic})
    
    return {"poem": result}
