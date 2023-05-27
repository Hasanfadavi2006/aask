from ai_bricks.api import openai
import stats
import os

DEFAULT_USER = os.getenv('COMMUNITY_USER','')

def use_key(key):
	openai.use_key(key)

usage_stats = stats.get_stats(user=DEFAULT_USER)
def complete(text, **kw):
	model = kw.get('model','gpt-4.0-turbo')  # Change default model to GPT-4
	llm = openai.model(model)
	llm.config['pre_prompt'] = 'output only in raw text' # for chat models
	resp = llm.complete(text, **kw)
	resp['model'] = model
	return resp

def embedding(text, **kw):
	model = kw.get('model','text-embedding-ada-002')  # Update this if GPT-4 has a corresponding text-embedding model
	llm = openai.model(model)
	resp = llm.embed(text, **kw)
	resp['model'] = model
	return resp

def embeddings(texts, **kw):
	model = kw.get('model','text-embedding-ada-002')  # Update this if GPT-4 has a corresponding text-embedding model
	llm = openai.model(model)
	resp = llm.embed_many(texts, **kw)
	resp['model'] = model
	return resp

def get_community_usage_cost():
	data = usage_stats.get(f'usage:v4:[date]:{DEFAULT_USER}')
	used = 0.0
	used += 0.04   * data.get('total_tokens:gpt-4.0-turbo',0) / 1000  # Update this line for GPT-4
	# ... rest of the function ...
	return used
