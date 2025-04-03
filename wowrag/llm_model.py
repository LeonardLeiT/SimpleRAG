import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_deepseek import ChatDeepSeek

def get_openai_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['OPENAI_API_KEY']

openai.api_key = get_openai_key()

def get_completion(prompt, temperature=0.7, api_key = get_openai_key()):
    client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)

    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a SchwarzRag assistant"},
        {"role": "user", "content": prompt},
    ],
    temperature=temperature,
    stream=False
)
    return response.choices[0].message.content

def get_completion_token(messages, temperature=0.7, max_tokens=500, api_key = get_openai_key()):
    client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        content = response.choices[0].message.content
        
        token_dict = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        }
        
        return content, token_dict
    
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return None, None

def get_completion_steam(prompt, temperature=0.7, api_key = get_openai_key()):
    client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)

    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a SchwarzRag assistant"},
        {"role": "user", "content": prompt},
    ],
    temperature=temperature,
    stream=False
)
    return response.choices[0].message.content

def get_completion_stream(prompt, temperature=0.7, api_key = get_openai_key()):
    client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)

    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a SchwarzRag assistant"},
        {"role": "user", "content": prompt},
    ],
    temperature=temperature,
    stream=True
    )
    for chunk in response:
        print(chunk.choices[0].delta.content, end="")
    print('\n')

def llm_deepseek():
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=get_openai_key(),
        base_url="https://api.deepseek.com"
    )   
    return llm