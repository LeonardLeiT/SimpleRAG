from llm_model import llm_deepseek
from rag_model import MyEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

class ChatRAGChain:
    def __init__(self, persist_directory=None, 
                 rag_system_prompt=None,
                 non_rag_system_prompt=None,
                 condense_question_prompt=None,
                 llm_model=None,
                 embedding_model=None):
        # Initialize LLM
        self.llm = llm_model if llm_model is not None else llm_deepseek()
        
        # Store custom prompts
        self.rag_system_prompt = rag_system_prompt
        self.non_rag_system_prompt = non_rag_system_prompt
        self.condense_question_prompt = condense_question_prompt
        
        # Check if persist_directory exists and contains data
        self.use_rag = False
        if persist_directory and os.path.exists(persist_directory):
            try:
                # Try to initialize the vector store to check if it contains data
                embedding = embedding_model if embedding_model is not None else MyEmbeddings()
                vector_store = Chroma(
                    embedding_function=embedding,
                    persist_directory=persist_directory
                )
                # Check if the collection exists and has documents
                if vector_store._collection and vector_store._collection.count() > 0:
                    self.use_rag = True
                    self.embedding = embedding
                    self.vector_store = vector_store
            except Exception:
                # If any error occurs during initialization, don't use RAG
                pass
        
        if self.use_rag:
            # Initialize retriever
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            
            # Set system prompt for QA with RAG
            if self.rag_system_prompt is None:
                self.rag_system_prompt = (
                    "You are a professional Q&A assistant. Please use the retrieved context to answer the user's question. "
                    "If you don't know the answer, just say you don't know. Please keep your answers concise. "
                    "\n\n"
                    "{context}"
                )
            
            # Set system prompt for question condensation
            if self.condense_question_prompt is None:
                self.condense_question_system_template = (
                    "Please refine the user's latest question based on the chat history. "
                    "If the latest question doesn't need refinement, return the original question."
                )
            else:
                self.condense_question_system_template = self.condense_question_prompt
            
            # Create QA prompt template
            self.qa_prompt = ChatPromptTemplate([
                ("system", self.rag_system_prompt),
                ("human", "{input}"),
            ])
            
            # Create question condensation prompt template
            self.condense_question_prompt = ChatPromptTemplate([
                ("system", self.condense_question_system_template),
                ("human", "{input}"),
            ])
            
            # Create document retrieval branch
            self.retrieve_docs = RunnableBranch(
                # Branch 1: Direct retrieval when no chat history
                (lambda x: not x.get("chat_history", False), 
                 (lambda x: x["input"]) | self.retriever),
                # Branch 2: Condense question first when chat history exists
                self.condense_question_prompt | self.llm | StrOutputParser() | self.retriever,
            )
            
            # Create document combination function
            def combine_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs["context"])
            
            # Create QA chain with RAG
            self.qa_chain = (
                RunnablePassthrough.assign(context=combine_docs)
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Create complete chat chain with RAG
            self.chat_chain = RunnablePassthrough.assign(
                context=(lambda x: x) | self.retrieve_docs
            ).assign(answer=self.qa_chain)
        else:
            # Set system prompt for QA without RAG
            if self.non_rag_system_prompt is None:
                self.non_rag_system_prompt = (
                    "You are a professional Q&A assistant. Please answer the user's question based on the chat history. "
                    "If you don't know the answer, just say you don't know. Please keep your answers concise. "
                )
            
            # Create QA prompt template without RAG
            self.qa_prompt = ChatPromptTemplate([
                ("system", self.non_rag_system_prompt),
                ("human", "{chat_history}\nHuman: {input}\nAssistant:"),
            ])
            
            # Create simple chat chain without RAG
            self.chat_chain = self.qa_prompt | self.llm | StrOutputParser()
    
    def chat(self, question, chat_history=None, stream=False):
        """
        Conduct a conversation
        :param question: User's question
        :param chat_history: Chat history in format [(question1, answer1), (question2, answer2), ...]
        :param stream: Whether to stream the response
        :return: If stream=True, returns a generator yielding answer chunks. If stream=False, returns the complete answer.
        """
        if chat_history is None:
            chat_history = []
            
        # Build input based on whether RAG is used
        if self.use_rag:
            # Convert chat history to message list format for RAG
            chat_history_messages = []
            for q, a in chat_history:
                chat_history_messages.extend([
                    HumanMessage(content=q),
                    AIMessage(content=a)
                ])
            
            chain_input = {
                "input": question,
                "chat_history": chat_history_messages
            }
        else:
            # For non-RAG mode, include chat history in the input
            chat_history_text = ""
            for q, a in chat_history:
                chat_history_text += f"Human: {q}\nAssistant: {a}\n"
            
            chain_input = {
                "input": question,
                "chat_history": chat_history_text
            }
        
        if stream:
            # Stream the response
            for chunk in self.chat_chain.stream(chain_input):
                if "answer" in chunk:
                    yield chunk["answer"]
                else:
                    yield chunk
        else:
            # Get complete response
            result = self.chat_chain.invoke(chain_input)
            return result["answer"] if "answer" in result else result

# Usage example
if __name__ == "__main__":
    # Test without database (non-RAG mode)
    print("=== Testing without database (non-RAG mode) ===")
    chat_rag = ChatRAGChain()
    
    # Test single-turn conversation with streaming
    question = "what is the free energy criterion for SC? and what my name is?"
    print("Question:", question)
    print("Answer: ", end="", flush=True)
    for chunk in chat_rag.chat(question, stream=True):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
    print("\n")
    
    # Test multi-turn conversation with streaming
    chat_history = [
        ("My name is Leilei", "Nice to meet you, Leilei!"),
        ("what are its properties?", "SC has unique electrical properties."),
        ("My name is Leilei", "Nice to meet you, Leilei!"),
    ]
    follow_up_question = "what is the free energy criterion for SC? and what my name is?"
    print("Question:", follow_up_question)
    print("Answer: ", end="", flush=True)
    for chunk in chat_rag.chat(follow_up_question, chat_history, stream=True):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
    print("\n")
    
    # Test with database (RAG mode)
    print("\n=== Testing with database (RAG mode) ===")
    chat_rag_with_db = ChatRAGChain(persist_directory='database/chroma')
    
    # Test single-turn conversation with streaming
    question = "what is the free energy criterion for SC?"
    print("Question:", question)
    print("Answer: ", end="", flush=True)
    for chunk in chat_rag_with_db.chat(question, stream=True):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
    print("\n")
    
    # Test multi-turn conversation with streaming
    chat_history = [
        ("what is SC?", "SC is a type of material."),
        ("what are its properties?", "SC has unique electrical properties."),
    ]
    follow_up_question = "what is the free energy criterion for SC?"
    print("Question:", follow_up_question)
    print("Answer: ", end="", flush=True)
    for chunk in chat_rag_with_db.chat(follow_up_question, chat_history, stream=True):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
    print("\n")
    
    # Test with custom prompts
    print("\n=== Testing with custom prompts ===")
    custom_rag_prompt = (
        "你是一个专业的助手。请基于以下上下文回答问题：\n\n{context}\n\n"
        "如果不知道答案，请直接说不知道。请保持回答简洁。"
    )
    custom_non_rag_prompt = (
        "你是一个专业的助手。请基于对话历史回答问题。"
        "如果不知道答案，请直接说不知道。请保持回答简洁。"
    )
    custom_condense_prompt = "请基于对话历史优化用户的问题。如果不需要优化，请直接返回原问题。"
    
    chat_rag_custom = ChatRAGChain(
        persist_directory='database/chroma',
        rag_system_prompt=custom_rag_prompt,
        non_rag_system_prompt=custom_non_rag_prompt,
        condense_question_prompt=custom_condense_prompt
    )
    
    # Test custom prompts with streaming
    question = "what is the free energy criterion for SC?"
    print("Question:", question)
    print("Answer: ", end="", flush=True)
    for chunk in chat_rag_custom.chat(question, stream=True):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
    print("\n") 