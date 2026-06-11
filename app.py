import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

#loading api keys
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#loading pdf
pdf_path = "docs/research.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Loaded {len(pages)} pages from PDF")

#splitting texts into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

#creating embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})
print("Vector store built successfully!")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_retries=3
)

# Prompt 1: Rephrase the follow-up question into a standalone question
condense_question_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the conversation history and a follow-up question, \
rephrase the follow-up question to be a standalone question that contains \
all necessary context. Do not answer the question, just rephrase it."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Prompt 2: Answer the question using retrieved context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Use the following context \
to answer the question. If you don't know the answer from the context, \
say you don't know.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Helper to format retrieved chunks into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain 1 — Rephrases the follow-up question into a standalone question
condense_question_chain = (
    condense_question_prompt
    | llm
    | StrOutputParser()
)

# Chain 2 — Retrieves context and answers the standalone question
def contextualized_question(input: dict):
    if input.get("chat_history"):
        return condense_question_chain
    else:
        return input["question"]

retrieval_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(
            retriever.invoke(
                contextualized_question(x)
                if isinstance(contextualized_question(x), str)
                else condense_question_chain.invoke(x)
            )
        )
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

# Memory store — holds conversation history
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the chain with message history
conversational_chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Conversation loop
print("\nConversational RAG Chatbot ready! Type 'quit' to exit.\n")

session_id = "user_session_1"

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    
    answer = conversational_chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    print(f"\nAssistant: {answer}\n")
