#!/usr/bin/env python
# coding: utf-8

# In[1]:


from build_embedder import *
from build_vectordb import create_retriever
from build_llm import *


# In[ ]:


#
# This one is to be used with Streamlit
#

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
# Luigi's OracleVectorStore wrapper for LangChain
from oracle_vector_db_lc import OracleVectorStore


# In[2]:


from langchain.prompts import ChatPromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
my_conv_memory = ConversationBufferMemory()


# In[11]:


def print_response(my_dict):
    desired_order = ['question', 'generated_question', 'answer', 'chat_history', 'source_documents']
    for key in desired_order:
        logger.info(f"{key}:")
        # print(f"    {type(my_dict[key])}")
        if isinstance(my_dict[key], list): 
            for item in my_dict[key]:
                logger.info(f'{item}')
        else:
            logger.info(f"    {my_dict[key]}")


# In[4]:


#
# def: get_answer  from LLM
#
def get_answer(rag_chain, question):
    response = rag_chain.invoke(question)

    if DEBUG:
        # logger.debug(f"Question: {question}")
        # logger.debug("The response:")
        # logger.debug(response)
        print_response(response)

    return response


# In[ ]:


#
# Initialize_rag_chain
#
# to run it only once
@st.cache_resource
def create_rag_chain():
    logger.debug(f'#### ENTER create_rag_chain() function ####')

    # 1. Load embeddings model
    embedder = create_embedder()

    # 2. restore vectordb
    if VECTOR_STORE == "CHROMA":
        # restore persistant chromadb
        try:
            vectorstore = Chroma(persist_directory=f"./vectorstore/{VECTORDB_FOLDER}_chromadb", embedding_function=embedder)
            logger.info(f'{VECTOR_STORE} vectordb restored')
        except Exception as e:
            logger.exception("Exception occured")
            sys.exit(1)
    elif VECTOR_STORE == "ORACLE":
        # restore oracle vectordb
        try:
            vectorstore = OracleVectorStore(embedding_function=embedder.embed_query, verbose=True)
            logger.info(f'{VECTOR_STORE} vectordb restored')
        except Exception as e:
            logger.exception("Exception occured")
            sys.exit(1)
    else:
        print(f'Please check the vector store to be used as part of configuration')
        exit()

    # 3. Create a retriever
    retriever = create_retriever(vectorstore)

    # 4. Build the LLM
    llm = build_llm()
    
    # 5. Build prompt template
    template = """You are a helpful assistant who searches for answer by going through provided context below.     Answer the question as truthfully as possible using only the context provided.     If the answer is not contained within the context below, say "I don't know".     Do not end your answer with a question.
    Context: {context}
    Question: {question}
    """

    rag_prompt = ChatPromptTemplate.from_template(template)

    # 6. Build conversation memory
    logger.info("Initializing Conversation Buffer Memory")
    global my_conv_memory
    my_conv_memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key='question', output_key='answer',
        return_messages=True,
        verbose = True
    )

    # 7. Create a conversation chain
    logger.info("Initializing Conversation Retrieval Chain")
    qa = ConversationalRetrievalChain.from_llm(
        llm, 
        chain_type="stuff", 
        retriever=retriever, 
        memory=my_conv_memory,
        return_source_documents=True,
        return_generated_question=True,
        rephrase_question=False, 
        combine_docs_chain_kwargs={'prompt': rag_prompt}
    )


    logger.info("\nRAG Chain created successfully")

    logger.debug(f'#### EXIT create_rag_chain() function ####')
    return qa


# In[ ]:


# 
# Reset Conversation Buffer Memory
# 
def clear_conv_memory():
    logger.debug(f'#### ENTER clear_conv_memory() function ####')
    global my_conv_memory
    my_conv_memory.clear()
    logger.info(f'Conversation Memory cleared')
    logger.debug(f'#### EXIT clear_conv_memory() function ####')


# In[6]:


# qa = create_rag_chain()


# In[ ]:


# result = qa.invoke("what is machine learning?")
# print_response(result)


# In[ ]:


# result = qa.invoke("what are different steps needed to build it?")
# print_response(result)


# In[ ]:


# result = qa.invoke("what is oracle autonomous database?")
# print_response(result)


# In[ ]:


# result = qa.invoke("what are the different offerings that it provides?")
# print_response(result)


# In[12]:


# result = qa.invoke("Provide more details about Oracle Autonomous Data Warehouse")
# print_response(result)

