import os
from typing import Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.tools.file_management.read import ReadFileTool
import autogen
from openai import OpenAI
import pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from flask import Flask, request, jsonify

from dotenv import load_dotenv
load_dotenv()

pinecone_client = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QueryToolInput(BaseModel):
    user_input: str = Field()
    index_name: str = Field()

class QueryTool(BaseTool):
    name = "query_tool"
    description = "Use this tool when you need to query into database based on user input"
    args_schema: Type[BaseModel] = QueryToolInput

    def _run(self, user_input: str, index_name: str):
        text = user_input.replace("\n", " ")
        query_embedding = openai.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        ).data[0].embedding

        # index_name = "trd-mission"
        index = pinecone_client.Index(index_name)
        result = index.query(
            top_k=10,
            include_values=True,
            include_metadata=True,
            vector=query_embedding
        )

        data = ""
        for match in result["matches"]:
            data += match["metadata"]["text"] + "\n"
        return data

def generate_llm_config(tool):
    function_schema = {
        "name": tool.name.lower().replace(" ", "_"),
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    if tool.args is not None:
        function_schema["parameters"]["properties"] = tool.args

    return function_schema

query_tool = QueryTool()

llm_config = {
    "functions": [
        generate_llm_config(query_tool),
    ],
    "config_list": [{"model": "gpt-3.5-turbo", "api_key": os.getenv("OPENAI_API_KEY")}],
    "timeout": 120,
}

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Get user question from the request
        data_question = request.json.get('data', '')
        user_question = data_question['question']
        if not user_question:
            response = jsonify({
                "status": "failed",
                "error": "Question is required"
            })
            return response, 400
        
        user = autogen.UserProxyAgent(
            name="user",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,
            },
        )

        user.register_function(
            function_map={
                query_tool.name: query_tool._run,
            }   
        )

        ChatbotAgent = autogen.AssistantAgent(
            name="Chatbot Agent",
            system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
            llm_config=llm_config,
        )

        index_list = pinecone_client.list_indexes().names()
        processed_data = user.initiate_chat(
            ChatbotAgent,
            message=f"""
                Answer the following user question: '{user_question}'.
                You need to retrieve data from database pinecone by just passing the question to the query tool as arguments.
                You need to pass the pinecone index name to the query tool as an argument where the index name is chosen based on the user question.
                Provided index names: {index_list}
                Then prompt into llm with the user question and data retrieved from pinecone.""",
            llm_config=llm_config,
        )

        chat_history = processed_data.chat_history
        answer = ''
        for message in chat_history:
            role = message.get('role', '')
            content = message.get('content', '')
            if role == 'user' and content != 'TERMINATE':
                answer = content
                break
        
        response =jsonify({
            "status": "success",
            "data": answer,
        })

        

        return response
    
    except Exception as e:
        response = jsonify({
            "status": "failed",
            "error": str(e)
        })
        return response, 500
    
@app.route('/hello-world', methods=['GET'])
def hello_world():
    try:
        response =jsonify({
            "status": "success",
            "data": "hello world"
        })
        return response
    
    except Exception as e:
        response = jsonify({
            "status": "failed",
            "error": str(e)
        })
        return response, 500

if __name__ == "__main__":
    app.run(host=os.getenv("HOST"), port=os.getenv("PORT"))
