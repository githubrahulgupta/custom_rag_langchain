#!/usr/bin/env python
# coding: utf-8

# In[1]:


from config import *

logger.info(f"Documents Directory: {docs_dir}")
logger.info(f"Documents folder being read: {docs_dir}{separator}{docs_folder_name}")


# In[2]:


import re


# In[3]:


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter


# In[4]:


# 
# Capturing individual documents name from a particular directory
# 

def get_docs_name():
    logger.debug(f'#### ENTER get_docs_name() function ####')
    actual_docs_dir = f"{docs_dir}{separator}{docs_folder_name}{separator}"
    logger.debug(f'Reading documents from path:\n{actual_docs_dir}')

    docs_name = []

    for filename in os.listdir(actual_docs_dir):
        if filename.lower().endswith((".pdf", ".txt")):
            docs_name.append(filename)
            
    logger.info(f'Documents being processed:')
    for i, doc in enumerate(docs_name):
        logger.info(f'({i+1}) {doc}')
    
    logger.debug(f'#### EXIT get_docs_name() function ####')
    return docs_name


# In[5]:


# 
# Add extra metadata to each document
# 
def add_metadata(doc):
    logger.debug(f'#### ENTER add_metadata() function ####')

    # Get the directory separator for the platform
    # separator = os.path.sep
    # print(f'Folder separator used w.r.t OS: {separator}')
    start = doc.metadata["source"].rfind(separator)
    # print(start)
    # end = doc.metadata["source"].find('.pdf')
    # print(end)
    # print(f'Doc name: {doc.metadata["source"][start+1:end]}')
    doc.metadata["name"] = doc.metadata["source"][start+1:]
    key = "page"
    if key not in doc.metadata:
        # print(f"{key} does not exist in the document metadata")
        doc.metadata[key] = 0
    # print(doc.metadata)

    logger.debug(f'#### EXIT add_metadata() function ####')


# In[6]:


#
# load all documents inside a particular directory
#
#
def load_all_docs():
    logger.debug(f'#### ENTER load_all_docs() function ####')

    all_docs = []
    # docs_dir = f"./documents/{docs_folder_name}/"
    actual_docs_dir = f"{docs_dir}{separator}{docs_folder_name}{separator}"
    
    book_name = get_docs_name()

    for filename in os.listdir(actual_docs_dir):
        # Loading pdf docs
        if filename.lower().endswith(".pdf"):
            logger.info(f"Loading document: {filename}...")
            loader = PyPDFLoader(f'{actual_docs_dir}{separator}{filename}')
        # load txt files
        elif filename.lower().endswith(".txt"):
            logger.info(f"Loading file: {filename}...")
            loader = TextLoader(f'{actual_docs_dir}{separator}{filename}')
        else:
            logger.warning(f'No loader defined for this file: {filename}')
            continue
            
        
        # loader split in pages
        docs = loader.load()

        # looping to add extra metadata to each page
        logger.debug("Original documents loaded as-is")
        for i, doc in enumerate(docs):
            add_metadata(doc)
            if DEBUG:
                logger.debug(f'Doc {i+1} Page Content: {doc.page_content}')
                logger.debug(f'Doc {i+1} Metdata: {doc.metadata}')

        all_docs.extend(docs)

    logger.info(f"\nLoaded {len(all_docs)} docs...\n")
    # print(f'##### All docs ####:\n')
    # [print(f'{doc}\n') for doc in all_docs]

    logger.debug(f'#### EXIT load_all_docs() function ####')
    return all_docs


# In[7]:


#
# do some post processing on text
#
def post_process(splits):
    logger.debug(f'#### ENTER post_process() function ####')
    for split in splits:
        # replace newline with blank
        split.page_content = split.page_content.replace("\n", " ")
        split.page_content = re.sub("[^a-zA-Z0-9 \n\.]", " ", split.page_content)
        split.page_content = re.sub(r"\.{2,}", ".", split.page_content)
        # remove duplicate blank
        split.page_content = " ".join(split.page_content.split())

    logger.debug(f'#### EXIT post_process() function ####')
    return splits


# In[8]:


#
# Split pages in chunk
#
def split_in_chunks(all_docs):
    logger.debug(f'#### ENTER split_in_chunks() function ####')
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["(?<=\. )", "\n"],
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        length_function=len
    )

    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Splitted the document in {len(chunks)} chunks...")
    
    # some post processing on text
    chunks = post_process(chunks)
    if DEBUG:
        logger.debug('=============CHUNKED & POST PROCESSED DOCUMENTS=============')
        for i, item in enumerate(chunks):
            logger.debug(f'Chunk ({i+1}):\n {item}')

    non_empty_chunks = []
    for i, item in enumerate(chunks):
        if dict(item)['page_content'] != '':
            non_empty_chunks.append(item)
        
    logger.info(f"Number of non-empty chunks: {len(non_empty_chunks)}")

    logger.debug(f'#### EXIT split_in_chunks() function ####')
    return non_empty_chunks


# In[9]:


# 1. Load a list of pdf documents
# all_docs = load_all_docs()
# all_docs


# In[10]:


# 2. Split pages in chunks
# document_chunks = split_in_chunks(all_docs)
# document_chunks
# doc_list, metadata_list = docs_and_metadata(document_splits)


# In[12]:


# release_log_file()


# In[ ]:




