#!/usr/bin/env python
# coding: utf-8

# In[1]:


from config import *
import ast, array, shutil, sys

from dotenv import load_dotenv
_ = load_dotenv()
logger.debug(f'Environment file loaded')


# In[ ]:


from load_and_chunk import get_docs_name
# from build_embedder import *


# In[ ]:


from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

# Using OCI vectordb 
import oracledb

# Luigi's OracleVectorStore wrapper for LangChain
from oracle_vector_db_lc import OracleVectorStore


# In[ ]:


# 
# required for oracle vectordb
# 
def initialize_vectordb_tables(cursor):
    logger.debug(f'#### ENTER initialize_vectordb_tables() function ####')
    # Drop tables
    table = f'{VECTORDB_FOLDER}_CHUNKS'
    # print(f'table: {table}')
    query = f"""
    begin
        execute immediate 'drop table {table}';
        exception when others then if sqlcode <> -942 then raise; end if;
    end;"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    table = f'{VECTORDB_FOLDER}_DOCS'
    # print(f'table: {table}')
    query = f"""
    begin
        execute immediate 'drop table {table}';
        exception when others then if sqlcode <> -942 then raise; end if;
    end;"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    table = f'{VECTORDB_FOLDER}_VECTORS'
    # print(f'table: {table}')
    query = f"""
    begin
        execute immediate 'drop table {table}';
        exception when others then if sqlcode <> -942 then raise; end if;
    end;"""
    # print(f'query: {query}')
    cursor.execute(query)

    logger.info(f"\nAll {VECTORDB_FOLDER} tables in DATABASE SCHEMA {os.getenv('DB_USER')}: ")
    query = f"""SELECT table_name FROM all_tables WHERE owner = '{os.getenv('DB_USER')}' and table_name like '{VECTORDB_FOLDER}%'"""
    # print(f'all tables: {query}')
    cursor.execute(query)
    
    for row in cursor:
         logger.info(row)
    
    # create tables
    query = f"""
    create table {VECTORDB_FOLDER}_VECTORS (
        id VARCHAR2(64) NOT NULL,
        VEC VECTOR(1024, FLOAT64),
        primary key (id))"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    query = f"""
    create table {VECTORDB_FOLDER}_DOCS (
        ID NUMBER NOT NULL,
        NAME VARCHAR2(100) NOT NULL,
        PRIMARY KEY (ID)  )"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    query = f"""
    create table {VECTORDB_FOLDER}_CHUNKS 
        (ID VARCHAR2(64) NOT NULL,
        CHUNK CLOB,
        PAGE_NUM VARCHAR2(10),
        DOC_ID NUMBER,
        PRIMARY KEY ("ID"),
        CONSTRAINT fk_{VECTORDB_FOLDER}_doc
                FOREIGN KEY (DOC_ID)
                REFERENCES {VECTORDB_FOLDER}_DOCS (ID)
        )"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    logger.info("Oracle vectordb tables initialized...")
    logger.debug(f'#### EXIT initialize_vectordb_tables() function ####')
    


# In[ ]:


# 
# required for oracle vectordb
# with this function every book added to DB is registered with a unique id
# 
def register_docs(docs_name, connection):
    logger.debug(f'#### ENTER register_docs() function ####')
    logger.info(f"Registering documents to vectordb table {VECTORDB_FOLDER}_DOCS...")
    with connection.cursor() as cursor:
                
        # get the new key
        query = f"SELECT MAX(ID) FROM {VECTORDB_FOLDER}_DOCS"
        cursor.execute(query)

        # Fetch the result
        row = cursor.fetchone()

        if row[0] is not None:
            new_key = row[0] + 1
        else:
            new_key = 1

        # insert the record for the book
        query = f"INSERT INTO {VECTORDB_FOLDER}_DOCS (ID, NAME) VALUES (:1, :2)"

        # Execute the query with your values
        cursor.execute(query, [new_key, docs_name])

    logger.info(f"Completed registering documents inside Oracle vectordb")
    logger.debug(f'#### EXIT register_docs() function ####')
    return new_key


# In[ ]:


# 
# required for oracle vectordb
# 
def save_chunks_in_db(document_chunks, doc_id, docs_name, connection):
    logger.debug(f'#### ENTER save_chunks_in_db() function ####')
    tot_errors = 0
    
    chunk_id = [] 
    chunk_text = [] 
    
    document_splits_str = [str(item) for item in document_chunks]
    
    with connection.cursor() as cursor:
        logger.info("Saving chunks to DB...")
        cursor.setinputsizes(None, oracledb.DB_TYPE_CLOB)

        for i, chunk in enumerate(document_splits_str):
            # chunk_id = i+1
            chunk_id.append(i+1)

            chunk_text_start = chunk.find("page_content=")
            chunk_metadata_start = chunk.find("metadata=")

            chunk_content = chunk[chunk_text_start+13:chunk_metadata_start]
            chunk_text.append(chunk_content)
            
            chunk_metadata=chunk[chunk_metadata_start+9:]
            chunk_metadata = ast.literal_eval(chunk_metadata) # parses the input string as a Python literal structure, such as a string, list, tuple, or dictionary.

            chunk_doc_id = doc_id[docs_name.index(chunk_metadata["name"])]
            chunk_page_num = int(chunk_metadata["page"])+1
            
            try:
                query = f"insert into {VECTORDB_FOLDER}_CHUNKS (ID, CHUNK, PAGE_NUM, DOC_ID) values (:1, :2, :3, :4)"
                cursor.execute(query, [i+1, chunk_content, chunk_page_num, chunk_doc_id])
            except Exception as e:
                logger.exception("Exception occured in save chunks...")
                tot_errors += 1
            
        logger.info(f'No. of chunk ids created inside get_chunk_content(): {len(chunk_id)}')
        logger.info(f"Completed savings chunks inside oracle vectordb")
        logger.debug(f'#### EXIT save_chunks_in_db() function ####')
        return chunk_id, chunk_text


# In[ ]:


# 
# required for oracle vectordb
# 
def save_embeddings_in_db(embeddings, chunks_id, connection):
    logger.debug(f'#### ENTER save_embeddings_in_db() function ####')
    tot_errors = 0

    with connection.cursor() as cursor:
        logger.info("Saving embeddings to DB...")

        for id, vector in zip(chunks_id, embeddings):        
            # 'f' single precision 'd' double precision
            if EMBEDDINGS_BITS == 64:
                input_array = array.array("d", vector)
            else:
                # 32 bits
                input_array = array.array("f", vector)

            try:
                # insert single embedding
                query = f"insert into {VECTORDB_FOLDER}_VECTORS values (:1, :2)"
                cursor.execute(query, [id, input_array])
            except Exception as e:
                logger.exception("Error in save embeddings...")
                tot_errors += 1

    logger.info(f"Total no. of errors in save_embeddings: {tot_errors}")
    logger.info(f"Completed savings embeddings inside oracle vectordb")
    logger.debug(f'#### EXIT save_embeddings_in_db() function ####')


# In[ ]:


#
# create vector store
#
def create_vectordb(VECTOR_STORE, document_chunks, embedder):
    logger.debug(f'#### ENTER create_vectordb() function ####')
    logger.info(f"Using {VECTOR_STORE} as Vector Store...")

    if VECTOR_STORE == "CHROMA":
        
        # in-memory chromadb
        if VECTOR_STORE_TYPE == 'in-memory':
            logger.info(f'{VECTOR_STORE_TYPE} vector store being created')
            try:
                vectorstore = Chroma.from_documents(
                    documents=document_chunks, 
                    embedding=embedder
                )
            except Exception as e:
                logger.exception("Exception occurred")
                sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred

        # persistant chromadb
        elif VECTOR_STORE_TYPE == 'persistent':
            logger.info(f'{VECTOR_STORE_TYPE} vector store being created')
    #         vectorstore = Chroma.from_documents(
    #             documents=document_chunks, 
    #             embedding=embedder, 
    #             persist_directory="./chroma_db"
    #         )
        
            # OCI GenAI Cohere Embedding supports a size of [1:96] as input array
            # Below code takes into consideration input array size limit
            vectordb_path = f"./vectorstore/{VECTORDB_FOLDER}_chromadb/" 
            if os.path.isdir(vectordb_path):
                shutil.rmtree(vectordb_path) 
                logger.info(f'Directory called {VECTORDB_FOLDER}_chromadb has been deleted from vectorstore folder')

            logger.info(f'A new directory called {VECTORDB_FOLDER}_chromadb will be created under vectorstore folder')            
            vectorstore = Chroma(
                            persist_directory=vectordb_path,
                            embedding_function=embedder)
            
            ids_list = [str(pos+1) for pos, s in enumerate(document_chunks)]
            logger.info(f'No. of document splits: {len(ids_list)}')
            
            logger.info(f'Document embeddings being created in a batch size of {EMBED_BATCH_SIZE} docs at a time')

            start=0
            while start < len(document_chunks):
                try:
                    vectorstore.add_documents(
                    ids = ids_list[start:start+EMBED_BATCH_SIZE],
                    documents = document_chunks[start:start+EMBED_BATCH_SIZE]
                    )
                    start+=EMBED_BATCH_SIZE
                except Exception as e:
                    logger.exception('Exception occured')
                    # print(f'\nERROR OCCURRED WHILE CREATING VECTOR DB:\n {error} ')
                    logger.debug(f'\nStart = {start}, End = {start+EMBED_BATCH_SIZE}')
                    for i, item in enumerate(document_chunks[start:start+EMBED_BATCH_SIZE]):
                        logger.debug(f'ID# : {ids_list[start+i-1]}')
                        logger.debug(f'Document: {item}')
                    # break
                    sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred
            
            # restore persistant chromadb
            vectorstore = Chroma(persist_directory=f"./vectorstore/{VECTORDB_FOLDER}_chromadb", embedding_function=embedder)
        
    elif VECTOR_STORE == "FAISS":
        try:
            vectorstore = FAISS.from_documents(
                documents=document_chunks, 
                embedding=embedder
            )
        except Exception as e:
            logger.exception("Exception occured")
            sys.exit(1)
        
    elif VECTOR_STORE == "ORACLE":
        # connect to db
        logger.info("Connecting to Oracle DB...")

        DSN = os.getenv('DB_HOST_IP') + "/" + os.getenv('DB_SERVICE')

        with oracledb.connect(user=os.getenv('DB_USER'), password=os.getenv('DB_PWD'), dsn=DSN) as connection:
            logger.info("Successfully connected to Oracle Database...")
            
            initialize_vectordb_tables(connection.cursor())
            
            # determine doc_id and save in table {VECTORDB_FOLDER}_DOCS
            docs_name = get_docs_name()
            doc_id = [register_docs(doc, connection) for doc in docs_name]
            # book_id = register_book(docs_name, connection)

            chunk_id, chunk_text = save_chunks_in_db(document_chunks, doc_id, docs_name, connection)
            
            logger.info(f'Document embeddings being created in a batch size of {EMBED_BATCH_SIZE} docs at a time')

            start=0
            embeddings = []
            while start < len(document_chunks):
                try:
                    chunk_embeddings = embedder.embed_documents(chunk_text[start:start+EMBED_BATCH_SIZE])
                    embeddings.extend(chunk_embeddings)
                    start+=EMBED_BATCH_SIZE
                except Exception as e:
                    logger.exception(f'\nERROR OCCURRED WHILE CREATING ORACLE VECTOR DB')
                    logger.debug(f'\nStart = {start}, End = {start+EMBED_BATCH_SIZE}')
                    for i, item in enumerate(document_chunks[start:start+EMBED_BATCH_SIZE]):
                        logger.debug(f'ID# : {ids_list[start+i-1]}')
                        logger.debug(f'Document: {item}')
                    # break
                    sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred
            logger.info(f'Number of embeddings created: {len(embeddings)}')

            # store embeddings
            # here we save in DB
            save_embeddings_in_db(embeddings, chunk_id, connection)

            # a txn is a document
            connection.commit()       

        # restore oracle vectordb
        # OracleVectorStore is custom class that inherits from langchain_core.vectorstores
        vectorstore = OracleVectorStore(embedding_function=embedder.embed_query, verbose=True)
    logger.debug(f'#### EXIT create_vectordb() function ####')
    
    return vectorstore


# In[ ]:


#
# create retrievere with optional reranker
#
def create_retriever(vectorstore):
    logger.debug(f'#### ENTER create_retriever() function ####')
    if ADD_RERANKER == False:
        # no reranking
        logger.info(f"No reranking...")
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        except Exception as e:
            logger.exception("Exception occured")
            sys.exit(1)
        # print output here
        
    else:
        # to add reranking
        logger.info("Adding reranking to QA chain...")

        # compressor = CohereRerank(cohere_api_key=os.getenv('COHERE_API_KEY'))

        # base_retriever = vectorstore.as_retriever(
        #     search_kwargs={"k": top_k}
        # )

        # retriever = ContextualCompressionRetriever(
        #     base_compressor=compressor, base_retriever=base_retriever
        # )
    logger.debug(f'#### EXIT create_retriever() function ####')
    return retriever


# In[ ]:


# 1. Load a list of pdf documents
# all_docs = load_all_docs()
# all_docs


# In[ ]:


# 2. Split pages in chunks
# document_chunks = split_in_chunks(all_docs)
# document_chunks
# doc_list, metadata_list = docs_and_metadata(document_splits)


# In[ ]:


# 3. Load embeddings model
# embedder = create_embedder()
# embedder


# In[ ]:


# 4. Create a Vectore Store and store embeddings
# vectorstore = create_vectordb(VECTOR_STORE, document_chunks, embedder)
# vectorstore


# In[ ]:


# vectorstore.similarity_search('what is data science?')


# In[ ]:


# release_log_file()


# In[ ]:




