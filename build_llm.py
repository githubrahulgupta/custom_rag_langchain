#!/usr/bin/env python
# coding: utf-8

# In[21]:


from config import *

import shutil, sys

from dotenv import load_dotenv
_ = load_dotenv()
logger.debug(f'Environment file loaded')


# In[22]:


from langchain_community.llms import OCIGenAI
# from langchain_community.llms import Cohere
from langchain_cohere.llms import Cohere


# In[23]:


# Custom LLM langchain class created by RG
from aqua_cllm import *


# In[24]:


#
# Build LLM
#
def build_llm():
    logger.debug(f'#### ENTER build_llm() function ####')
    logger.info(f"Using {LLM_TYPE} {LLM_NAME} model...")

    if LLM_TYPE == "OCI_GEN_AI":
#       oci_config = load_oci_config()
        try:
            llm = OCIGenAI(
                model_id=LLM_NAME, 
                service_endpoint=os.getenv('OCI_GENAI_ENDPOINT'),
                compartment_id=os.getenv('COMPARTMENT_ID'),
                #temperature=TEMPERATURE # old endpoint
                model_kwargs={"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}#, "stop": ["populous"]} # new endpoint
            )
        except Exception as e:
            logger.exception("Exception occurred")
            sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred
    
    elif LLM_TYPE == "COHERE_OPEN_SOURCE":
        try:
            llm = Cohere(
                # model="command",  # using large model and not nightly
                cohere_api_key=os.getenv('COHERE_API_KEY'),
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
        except Exception as e:
            logger.exception("Exception occurred")
            sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred
        
    elif LLM_TYPE == "OCI_DS_AQUA":
        config = oci.config.from_file("~/.oci/config") # replace with the location of your oci config file
        try:
            llm = DS_AQUA_LLM(
                    temperature=TEMPERATURE, 
                    config=config, 
                    # compartment_id=compartment_id, 
                    service_endpoint=os.getenv('OCI_AQUA_ENDPOINT'),        
                    max_tokens = MAX_TOKENS, 
                    top_k = top_k, 
                    top_p = 1
                    )
            logger.info(f'Config used for AQUA LLM:\n{llm}')
        except Exception as e:
            logger.exception("Exception occurred")
            sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred

    logger.debug(f'#### EXIT build_llm() function ####')
    return llm


# In[25]:


# 6. Build the LLM
# llm = build_llm()


# In[26]:


# llm


# In[27]:


# llm.invoke("Tell me one fact about earth")


# In[ ]:




