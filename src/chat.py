import logging
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from src.config import logger, api_status
from src import app_state


def get_chat_model(model_name: str = "huggingface"):
    """Get chat model based on selection
    
    Args:
        model_name: Name of the chat model to use
        
    Returns:
        Language model instance
    """
    # Handle dict input from Gradio dropdown
    if isinstance(model_name, dict):
        model_name = model_name.get("value", "huggingface")
        
    if model_name == "openai" and api_status["openai"]:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
    elif model_name == "groq" and api_status["groq"]:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.7
        )
    else:
        # Default to HuggingFace with explicitly specified model
        from langchain_community.llms import HuggingFaceHub
        return HuggingFaceHub(
            repo_id="google/flan-t5-xl",
            model_kwargs={"temperature": 0.7, "max_length": 512},
            task="text2text-generation"  # Explicitly specify the task
        )

def setup_qa_chain(vectordb, model_name: str = "huggingface"):
    """Setup question answering chain
    
    Args:
        vectordb: Vector database instance
        model_name: Name of the chat model to use
        
    Returns:
        ConversationalRetrievalChain instance
    """
    # Получаем модель
    llm = get_chat_model(model_name)
    
    # Создаем память для истории разговора
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",  # Явно указываем ключ вывода для сохранения в память
        return_messages=True
    )
    
    # Шаблоны подсказок
    condense_question_prompt = PromptTemplate.from_template(
        """Given the following conversation and a follow up question, rephrase the follow up question 
        to be a standalone question that captures all relevant context from the conversation.
        
        Chat History:
        {chat_history}
        
        Follow Up Input: {question}
        Standalone question:"""
    )
    
    qa_prompt = PromptTemplate.from_template(
        """You are an assistant that helps users understand YouTube video content based on its subtitles.
        Answer the question based on the following context from the video subtitles.
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a concise and helpful answer. If the answer is not in the context, say so.
        Include relevant timestamps if available in the context.
        
        Answer:"""
    )
    
    # Создаем QA цепочку
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        return qa_chain
    except Exception as e:
        logger.error(f"Error setting up QA chain: {e}")
        # Если возникла ошибка, выводим информативное сообщение и пытаемся использовать более простую модель
        logger.info("Falling back to simpler model configuration")
        
        # Пробуем использовать более простую модель HuggingFace
        from langchain_community.llms import HuggingFaceHub
        simple_llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",  # Более простая и надежная модель
            model_kwargs={"temperature": 0.5, "max_length": 256},
            task="text2text-generation"
        )
        
        # Создаем более простую версию QA цепочки
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=simple_llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
        )
        
        return qa_chain

def chat_with_subtitles(message: str, history: List, model_name: str = "huggingface"):
    """Chat with the video content
    
    Args:
        message: User message
        history: Chat history
        model_name: Name of the chat model
        
    Returns:
        Updated history with bot response
    """
    if not hasattr(app_state, 'qa_chain') or not app_state.qa_chain:
        return history + [[message, "Please process a video first before chatting."]]
    
    if not message:
        return history
    
    try:
        # Handle dict input from Gradio dropdown
        if isinstance(model_name, dict):
            model_name = model_name.get("value", "huggingface")
        
        # Initialize current_model if it doesn't exist
        if not hasattr(app_state, 'current_model'):
            app_state.current_model = "huggingface"
        
        # Update model if needed
        if model_name != app_state.current_model:
            app_state.qa_chain = setup_qa_chain(app_state.vectordb, model_name)
            app_state.current_model = model_name
        
        # Get response
        result = app_state.qa_chain({"question": message})
        answer = result.get("answer", "")
        
        # Fetch source timestamps if available
        source_docs = result.get("source_documents", [])
        if source_docs:
            timestamps = []
            for doc in source_docs:
                if "time_str" in doc.metadata:
                    timestamps.append(doc.metadata["time_str"])
            
            if timestamps:
                answer += f"\n\nRelevant timestamps: {', '.join(timestamps)}"
        
        # Update history
        return history + [[message, answer]]
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return history + [[message, f"Error: {str(e)}"]]
