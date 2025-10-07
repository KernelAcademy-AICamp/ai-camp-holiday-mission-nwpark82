import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from openai import OpenAI


load_dotenv('env.txt')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


#문서 데이터 로드 및 전처리

loader = TextLoader('./res/생성형AI와일자리.txt', encoding='utf-8')
data = loader.load()
text = data[0].page_content

text_splitter = RecursiveCharacterTextSplitter(
     chunk_size= 500,  # 각 chunk의 최대 길이
    chunk_overlap=50,
    length_function=len
)

texts = text_splitter.split_text(text)
print(texts)

#문서 임베딩 생성
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

print(embeddings)

#벡터DB구축(Faiss/ChromaDB등)

knowledge_base = FAISS.from_texts(texts, embeddings)
print(knowledge_base)

question = "생성형 AI가 사람의 일자리를 얼마나 대체할까?"
references  = knowledge_base.similarity_search(question)
print(len(references))
print(references)

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

#질문에 대한 관련 문서 검색

chain = load_qa_chain(llm, chain_type = "stuff")

#검색 결과 기반 답변 생성
with get_openai_callback() as cb:
    response = chain.run(input_documents=references, question = question)
    print(cb)

print(response)