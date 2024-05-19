import os, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

####################################
# GLOBAL VARIABLES CONFIGURATION
####################################
poc_name = 'coe_demo'

DEBUG = False

start_datetime = datetime.now()

current_dir = os.getcwd()

# Get the directory separator for the Operating System on which script is run
separator = os.path.sep

docs_dir = f"{current_dir}{separator}documents"
docs_folder_name = poc_name

logs_dir = f"{current_dir}{separator}logs"
log_file = f'{logs_dir}{separator}{docs_folder_name}_{start_datetime.strftime("%Y_%m_%d_%H_%M_%S")}.log'


####################################
# LOG LEVEL CONFIGURATIONS
####################################

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
# f_handler = logging.FileHandler(log_file, mode='w')
# Create a RotatingFileHandler with a maximum file size of 1MB and keep up to 3 backup files
f_handler = RotatingFileHandler(log_file, mode='w', maxBytes=1024*1024, backupCount=3)
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.info(f"Current Working Directory: {current_dir}")
logger.info(f'Folder separator used w.r.t OS: {separator}')
logger.info(f"Log File: {log_file}")

# Function to release the log file
def release_log_file():
    logger.info(f'#### ENTER release_log_file() function ####')

    # Get the file handler
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break
    else:
        return

    logger.info(f'Releasing log file...')
    # Close the file handler
    file_handler.close()

    # Remove the file handler from the logger
    logger.removeHandler(file_handler)

    print("Log file released successfully.")

####################################
# CHUNK CONFIGURATIONS
####################################
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50

####################################
# EMBEDDINGS CONFIGURATION
####################################
CACHE_EMBEDDINGS = False
CACHE_EMBEDDINGS_FOLDER = poc_name

EMBED_TYPE = "cohere_oci"
EMBED_MODEL_NAME = "cohere.embed-english-v3.0" # cohere.embed-english-light-v2.0, cohere.embed-english-light-v3.0, cohere.embed-english-v3.0

# EMBED_TYPE='cohere_open_source'
# EMBED_MODEL_NAME = "embed-english-v3.0"

# EMBED_TYPE = "hugging_face"
# see: https://huggingface.co/spaces/mteb/leaderboard
# see also: https://github.com/FlagOpen/FlagEmbedding
# base seems to work better than small
# EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"


####################################
# VECTOR STORE CONFIGURATION
####################################
#
# Vector Store (Chrome or FAISS)
#
# VECTOR_STORE = "FAISS"
VECTOR_STORE = "CHROMA"
# VECTOR_STORE = "ORACLE"

VECTOR_STORE_TYPE = 'persistent' # persistent, in-memory
VECTORDB_FOLDER = poc_name # only in case of persistent vectordb

EMBED_BATCH_SIZE = 96

# bits used to store embeddings
# possible values: 32 or 64
# must be aligned with the create_tables.sql used
EMBEDDINGS_BITS = 64 # specific to oracle vectordb

# ID generation: LLINDEX, HASH, BOOK_PAGE_NUM
# define the method to generate ID
ID_GEN_METHOD = "HASH" # specific to oracle vectordb

# number of docs to return from Retriever
top_k = 3

# to add Cohere reranker to the QA chain
ADD_RERANKER = False


####################################
# LLM CONFIGURATION
####################################
#
# LLM Config
#
# LLM_TYPE = "COHERE_OPEN_SOURCE"
LLM_TYPE = "OCI_GEN_AI"
# LLM_TYPE = "OCI_DS_AQUA"

# 
# type of LLM Model. The choice has been parametrized
# # cohere.command, cohere.command-light, meta.llama-2-70b-chat
LLM_NAME = "cohere.command"
# LLM_NAME = "Mistral-7B-Instruct-v0.2" # for OCI DS AQUA

# max tokens returned from LLM for single query
MAX_TOKENS = 1500
# to avoid "creativity"
TEMPERATURE = 0

#
# OCI GenAI configs
#
TIMEOUT = 30
