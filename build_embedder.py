#!/usr/bin/env python
# coding: utf-8

# In[1]:


from config import *

import shutil, sys

from dotenv import load_dotenv
_ = load_dotenv()
logger.debug(f'Environment file loaded')


# In[ ]:


# for caching embeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

# from langchain_community.embeddings import CohereEmbeddings
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OCIGenAIEmbeddings


# In[ ]:


#
# Load the embedding model
#
def create_embedder():
    logger.debug(f'#### ENTER create_embedder() function ####')

    # Introduced to cache embeddings and make it faster
    if CACHE_EMBEDDINGS:
        logger.info(f'Cache Embeddings enabled')
        cache_embeddings_path = f"./embeddings_cache/{CACHE_EMBEDDINGS_FOLDER}/" 

        if os.path.isdir(cache_embeddings_path):
            shutil.rmtree(cache_embeddings_path) 
            logger.info(f'Directory called {CACHE_EMBEDDINGS_FOLDER} has been deleted from embeddings_cache folder')
        else:
            logger.info(f'A new directory called {CACHE_EMBEDDINGS_FOLDER} will be created under embeddings_cache folder')

        fs = LocalFileStore(cache_embeddings_path)
        
    if EMBED_TYPE == "cohere_open_source":
        logger.info(f"Loading Cohere Open Source Model {EMBED_MODEL_NAME}")
        try:
            embedder = CohereEmbeddings(
                model=EMBED_MODEL_NAME, 
                cohere_api_key=os.getenv('COHERE_API_KEY')
            )
        except Exception as e:
            logger.exception("Exception occurred")
            sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred

    elif EMBED_TYPE == "hugging_face":
        logger.info(f"Loading Hugging Face Embeddings Model: {EMBED_MODEL_NAME}")

        model_kwargs = {"device": "cpu"}
        # changed to True for BAAI, to use cosine similarity
        encode_kwargs = {"normalize_embeddings": True}
        try:
            embedder = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        except Exception as e:
            logger.exception("Exception occurred")
            sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred

    elif EMBED_TYPE == "cohere_oci":
        logger.info(f"Loading OCI GenAI Cohere Embeddings Model: {EMBED_MODEL_NAME}")
        try:
            embedder = OCIGenAIEmbeddings(
                model_id=EMBED_MODEL_NAME, 
                service_endpoint=os.getenv('OCI_GENAI_ENDPOINT'),
                compartment_id=os.getenv('COMPARTMENT_ID'), 
                truncate = 'NONE'
                )
        except Exception as e:
            logger.exception("Exception occurred")
            sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred

    # the cache for embeddings
    if CACHE_EMBEDDINGS:
        try:
            cached_embedder = CacheBackedEmbeddings.from_bytes_store(
                # embed_model, fs, namespace=embed_model.model_name # HUGGING FACE
                # embed_model, fs, namespace=embed_model.model # COHERE
                embedder, fs, namespace=embedder.model_id # OCI GEN AI COHERE
            )
            logger.debug(f'#### EXIT create_embedder() function ####')
        except Exception as e:
            logger.exception('Exception occured')
            sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred
        return cached_embedder
    else:
        logger.debug(f'#### EXIT create_embedder() function ####')
        return embedder


# In[ ]:


# 3. Load embeddings model
# embedder = create_embedder()


# In[ ]:


# release_log_file()

