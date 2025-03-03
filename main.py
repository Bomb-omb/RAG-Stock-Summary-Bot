from query_handling import handle_user_query
from transformers import AutoTokenizer, MistralForCausalLM
from dotenv import load_dotenv
import os
import requests
from langchain_huggingface import HuggingFaceEndpoint
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")