#!/usr/bin/env python
# coding: utf-8

# In[1]:


from load_and_chunk import *
from build_embedder import *
from build_vectordb import *


# In[ ]:


# 1. Load a list of documents
all_pages = load_all_docs()
# all_pages

# 2. Split pages in chunks
document_splits = split_in_chunks(all_pages)

# 3. Load embeddings model
embedder = create_embedder()

# 4. Create a Vectore Store and store embeddings
# for in-memory chromadb
# vectorstore = create_vector_store(VECTOR_STORE_NAME, document_splits, embedder)

# for persistant chromadb & OCI vectordb
create_vectordb(VECTOR_STORE, document_splits, embedder)

