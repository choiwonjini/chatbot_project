from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from apikey import API_KEY
import os

# --- 초기 설정 (앱 실행 시 한 번만 수행) ---

# OpenAI API 키 설정
os.environ['OPENAI_API_KEY'] = API_KEY

# 텍스트 파일 로드
documents = TextLoader('C:\\Users\\32217778\\공부\\chatbot\\chatbot_project\\분리배출_가이드라인.txt', encoding='utf-8').load()

# 텍스트 분할 (Chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# 임베딩 모델 설정
embeddings = SentenceTransformerEmbeddings(model_name='jhgan/ko-sbert-nli') # 한국어 특화 모델

# 벡터 DB 생성
db = Chroma.from_documents(docs, embeddings)

# LLM 및 QA 체인 생성
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
chain = load_qa_chain(llm, chain_type='stuff')


# --- 답변 생성 함수 (쿼리가 들어올 때마다 호출) ---

def get_rag_answer(query):
    """
    사용자의 질문을 받아 RAG 파이프라인을 통해 답변을 생성하는 함수
    """
    # 1. 유사도 검색으로 관련 문서 찾기
    matching_docs = db.similarity_search(query)
    
    # 2. QA 체인을 실행하여 답변 생성
    answer = chain.run(input_documents=matching_docs, question=query)
    
    return answer
