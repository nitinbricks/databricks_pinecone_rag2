# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-nitin_wagh` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0912-181252-vaeks2ps/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # 3/ Creating the Chat bot with Retrieval Augmented Generation (RAG)
# MAGIC
# MAGIC <img src="https://github.com/nitinbricks/databricks_pinecone_rag/blob/main/Databricks_Pinecone_RAG_architecture.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">
# MAGIC
# MAGIC
# MAGIC Our Vector Search Index is now ready!
# MAGIC
# MAGIC Let's now create and deploy a new Model Serving Endpoint to perform RAG.
# MAGIC
# MAGIC The flow will be the following:
# MAGIC
# MAGIC - A user asks a question
# MAGIC - The question is sent to our serverless Chatbot RAG endpoint
# MAGIC - The endpoint searches for docs similar to the question, leveraging Pinecone Vector Search on our Documentation table
# MAGIC - The endpoint creates a prompt enriched with the doc
# MAGIC - The prompt is sent to the AI Gateway, ensuring security, stability and governance
# MAGIC - The gateway sends the prompt to a MosaicML LLM Endpoint (currently LLama 2 70B)
# MAGIC - Mosaic returns the result
# MAGIC - We display the output to our customer!

# COMMAND ----------

# MAGIC %pip install mlflow[gateway] databricks-sdk==0.8.0 pinecone-client
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=dbdemos $db=chatbot $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Deploying an AI gateway to Mosaic ML Endpoint
# MAGIC
# MAGIC <!-- <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-inference-1.png?raw=true" style="float: right; margin-left: 10px"  width="600px;"> -->
# MAGIC
# MAGIC
# MAGIC With MLFlow, Databricks introduced the concept of AI Gateway ([documentation](https://mlflow.org/docs/latest/gateway/index.html)).
# MAGIC
# MAGIC AI Gateway acts as a proxy between your application and LLM APIs. It offers:
# MAGIC
# MAGIC - API key management
# MAGIC - Unified access point to easily switch the LLM backend without having to change your implementation
# MAGIC - Throughput control
# MAGIC - Logging and retries
# MAGIC
# MAGIC *Note: if you don't have a MosaicML key, you can also deploy an OpenAI gateway route:*
# MAGIC
# MAGIC ```
# MAGIC     gateway.create_route(
# MAGIC         name=openai_route_name,
# MAGIC         route_type="llm/v1/completions",
# MAGIC         model={
# MAGIC             "name": "gpt-35-turbo",
# MAGIC             "provider": "openai",
# MAGIC             "openai_config": {
# MAGIC                 "openai_api_type": "azure",
# MAGIC                 "openai_deployment_name": "dbdemo-gpt35",
# MAGIC                 "openai_api_key": dbutils.secrets.get(scope="dbdemos", key="mosaic_ml_api_key"),
# MAGIC                 "openai_api_base": "https://dbdemos-open-ai.openai.azure.com/",
# MAGIC                 "openai_api_version": "2023-03-15-preview"
# MAGIC             }
# MAGIC         }
# MAGIC     )
# MAGIC ```

# COMMAND ----------

#init MLflow experiment
import mlflow
from mlflow import gateway
init_experiment_for_batch("llm-chatbot-rag", "rag-model")

gateway.set_gateway_uri(gateway_uri="databricks")
#define our embedding route name, this is the endpoint we'll call for our embeddings
mosaic_route_name = "mosaicml-llama2-70b-completions"
openai_embedding_name = "openai_text_embedding_ada_002"

try:
    routes = gateway.search_routes()
except:
    #Temp try/catch when no route exist, see mlfow #9385
    routes = []
route_exists = any(route for route in routes if route.name == mosaic_route_name)

if not route_exists:
    # Create a Route for embeddings with Azure OpenAI
    print(f"Creating the route {mosaic_route_name}")
    print(gateway.create_route(
        name=mosaic_route_name,
        route_type="llm/v1/completions",
        model={
            "name": "llama2-70b-chat",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="dbdemos", key="mosaic_ml_api_key")
            }
        }
    ))

open_route_exists = any(route for route in routes if route.name == openai_embedding_name)

if not open_route_exists:
    print(gateway.create_route(
        name=openai_embedding_name,
        route_type="llm/v1/embeddings",
        model={
            "name": "text-embedding-ada-002",
            "provider": "openai",
            "openai_config": {
                "openai_api_key": dbutils.secrets.get(scope="dbdemos", key="openai")
            }
        }
    ))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's try our AI Gateway:
# MAGIC
# MAGIC AI Gateway accepts Databricks tokens as its authentication mechanism. 
# MAGIC
# MAGIC Let's send a simple REST call to our gateway. Note that we don't specify the LLM key nor the model details, only the gateway route.

# COMMAND ----------

from mlflow import gateway
print(f"calling AI gateway {gateway.get_route(mosaic_route_name).route_url}")

r = mlflow.gateway.query(route="mosaicml-llama2-70b-completions", data={"prompt": "What is Databricks Lakehouse?"})

print(r)

e = mlflow.gateway.query(route="openai_text_embedding_ada_002",data={"text": "test sentence"})

print(e)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Let's now create an endpoint for the RAG chatbot, using the gateway we deployed
# MAGIC
# MAGIC <!-- <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-inference-1.png?raw=true" style="float: right; margin-left: 10px"  width="600px;"> -->
# MAGIC
# MAGIC Our gateway is ready, and our different model deployements can now securely use the MosaicML route to query our LLM.
# MAGIC
# MAGIC We are now going to build our Chatbot RAG model and deploy it as an endpoint for realtime Q&A!
# MAGIC
# MAGIC #### A note on prompt engineering
# MAGIC
# MAGIC The usual prompt engineering method applies for this chatbot. Make sure you're prompting your model with proper parameters and matching the model prompt format if any.
# MAGIC
# MAGIC For a production-grade example, you'd typically use `langchain` and potentially send the entire chat history to your endpoint to support "follow-up" style questions.
# MAGIC
# MAGIC More advanced chatbot behavior can be added here, including Chain of Thought, history summarization etc.
# MAGIC
# MAGIC Here is an example with `langchain`:
# MAGIC
# MAGIC ```
# MAGIC from langchain.llms import MlflowAIGateway
# MAGIC
# MAGIC gateway = MlflowAIGateway(
# MAGIC     gateway_uri="databricks",
# MAGIC     route="mosaicml-llama2-70b-completions",
# MAGIC     params={"temperature": 0.7, "top_p": 0.95,}
# MAGIC   )
# MAGIC prompt = PromptTemplate(input_variables=['context', 'question'], template=<your template as string>)
# MAGIC ```
# MAGIC
# MAGIC To keep our demo super simple and not getting confused with `langchain`, we will create a plain text template. 

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel
import pinecone      
import os

#Service principal Databricks PAT token we'll use to access our AI Gateway
os.environ['AI_GATEWAY_SP'] = dbutils.secrets.get("dbdemos", "ai_gateway_service_principal")
os.environ['PINECONE_API_KEY'] = dbutils.secrets.get("dbdemos", "pinecone_api_key")


route = gateway.get_route(mosaic_route_name).route_url
route_embeddings = gateway.get_route(openai_embedding_name).route_url

workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

class ChatbotRAG(mlflow.pyfunc.PythonModel):
      
    def create_embeddings(self,content):
        #Note the AI_GATEWAY_SP environement variable. It contains a service principal key
        response = requests.post(route_embeddings, json = {"text": content}, 
                                 headers={"Authorization": "Bearer "+os.environ['AI_GATEWAY_SP']})
        response.raise_for_status()
        response = response.json()
        return response['embeddings']

    def find_relevent_from_pinecone(self, questions, num_results=1, relevant_threshold = .66):
      pinecone.init(      
        api_key=os.environ['PINECONE_API_KEY'],    
        environment='gcp-starter'      
      )
      embeddings = self.create_embeddings(questions)

      dbdemos_index = pinecone.Index('dbdemos-index')
      results = dbdemos_index.query(
            vector=embeddings,
            top_k=1,
            include_metadata=True
          )
      
      if results is not None:
        metadata = results['matches'][0]['metadata']
        return {"url": metadata['url'], "content": metadata['content']}
      else: 
        return None


    def predict(self, context, model_input):
        import os
        import requests
        import numpy as np
        import pandas as pd
        # If the input is a DataFrame or numpy array,
        # convert the first column to a list of strings.
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, np.ndarray):
            model_input = model_input[:, 0].tolist()
        elif isinstance(model_input, str):
            model_input = [model_input]
        answers = []
        for question in model_input:
          #Build the prompt
          prompt = "[INST] <<SYS>>You are an assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, DW and platform, API or infrastructure administration question related to Databricks. If the question is not related to one of these topics, kindly decline to answer."
          #doc = self.find_relevant_doc(question)
          doc = self.find_relevent_from_pinecone(question)
          #Add docs from our knowledge base to the prompt
          if doc is not None:
            prompt += f"\n\n Here is a documentation page which might help you answer: \n\n{doc['content']}"
          #Final instructions
          prompt += f"\n\n <</SYS>>Answer the following user question. If you don't know or the question isn't relevant or professional, say so. Only give a detailed answer. Don't have note or comment.\n\n  Question: {question}[/INST]"
          #Note the AI_GATEWAY_SP environement variable. It contains a service principal key
          response = requests.post(route, json = {"prompt": prompt, "max_tokens": 500}, headers={"Authorization": "Bearer "+os.environ['AI_GATEWAY_SP']})
          response.raise_for_status()
          response = response.json()
          
          if 'candidates' not in response:
            raise Exception(f"Can't parse response: {response}")
          answer = response['candidates'][0]['text']
          if doc is not None:
            answer += f"""\nFor more details, <a href="{doc['url']}">open the documentation</a>  """
          answers.append({"answer": answer.replace('\n', '<br/>'), "prompt": prompt})
        return answers

# COMMAND ----------

# DBTITLE 1,Let's try our chatbot in the notebook directly:
proxy_model = ChatbotRAG()
results = proxy_model.predict(None, ["How can I track billing usage on my workspaces?"])
print(results[0]["answer"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving our chatbot model in Unity Catalog

# COMMAND ----------

from mlflow.models import infer_signature
with mlflow.start_run(run_name="nw_chatbot_rag") as run:
    chatbot = ChatbotRAG()
    #Let's try our model calling our Gateway API: 
    signature = infer_signature(["some", "data"], results)
    #Temp fix, do not use mlflow 2.6
    mlflow.pyfunc.log_model("model", python_model=chatbot, 
                            signature=signature, pip_requirements=["mlflow==2.4.0", "cloudpickle==2.0.0","pinecone-client"]) #
    #mlflow.set_tags({"route": proxy_model.route})
print(run.info.run_id)

# COMMAND ----------

#Enable Unity Catalog with mlflow registry
mlflow.set_registry_uri('databricks-uc')

client = MlflowClient()
try:
  #Get the model if it is already registered to avoid re-deploying the endpoint
  latest_model = client.get_model_version_by_alias(f"{catalog}.{db}.nw_dbdemos_chatbot_model", "prod")
  print(f"Our model is already deployed on UC: {catalog}.{db}.nw_dbdemos_chatbot_model")
except:  
  #Add model within our catalog
  latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/model', f"{catalog}.{db}.nw_dbdemos_chatbot_model")
  client.set_registered_model_alias(name=f"{catalog}.{db}.nw_dbdemos_chatbot_model", alias="prod", version=latest_model.version)

  #Make sure all other users can access the model for our demo(see _resource/00-init for details)
  set_model_permission(f"{catalog}.{db}.nw_dbdemos_chatbot_model", "ALL_PRIVILEGES", "account users")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Let's now deploy our realtime model endpoint
# MAGIC
# MAGIC Let's leverage Databricks Secrets to load the Service Principal key when the endpoint starts. See the [documentation](https://docs.databricks.com/en/machine-learning/model-serving/store-env-variable-model-serving.html) for more details 

# COMMAND ----------

#Helper for the endpoint rest api, see details in _resources/00-init
serving_client = EndpointApiClient()
#Start the enpoint using the REST API (you can do it using the UI directly)
serving_client.create_enpoint_if_not_exists("nw_dbdemos_chatbot_rag", 
                                            model_name=f"{catalog}.{db}.nw_dbdemos_chatbot_model", 
                                            model_version = latest_model.version, 
                                            workload_size="Small",
                                            scale_to_zero_enabled=True, 
                                            wait_start = True, 
                                            environment_vars={"AI_GATEWAY_SP": "{{secrets/dbdemos/ai_gateway_service_principal}}","PINECONE_API_KEY": "{{secrets/dbdemos/pinecone_api_key}}"})

# COMMAND ----------

# MAGIC %md
# MAGIC Our endpoint is now deployed! You can directly [open it from the UI](/endpoints/dbdemos_embedding_endpoint) and visualize its performance!
# MAGIC
# MAGIC Let's run a REST query to try it in Python. As you can see, we send the `test sentence` doc and it returns an embedding representing our document.

# COMMAND ----------

# DBTITLE 1,Let's try to send a query to our chatbot
import timeit
question = "How can I track billing usage on my workspaces?"

answer = requests.post(f"{serving_client.base_url}/realtime-inference/nw_dbdemos_chatbot_rag/invocations", 
                       json={"dataframe_split": {'data': [question]}}, 
                       headers=serving_client.headers).json()
#Note: If your workspace has ip access list, you need to allow your model serving endpoint to hit your AI gateways. Please reach out your Databrics Account team for IP ranges.
display_answer(question, answer['predictions'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Congratulations! You have deployed your first Gen AI RAG model!
# MAGIC
# MAGIC You're now ready to deploy the same logic for your internal knowledge base leveraging Lakehouse AI.
# MAGIC
# MAGIC We've seen how the Lakehouse AI is uniquely positioned to help you solve your Gen AI challenge:
# MAGIC
# MAGIC - Simplify Data Ingestion and preparation with Databricks Engineering Capabilities
# MAGIC - Stores Embeddings in scalable Pinecone Vector Index
# MAGIC - Simplify, secure and control your LLM access with AI gateway
# MAGIC - Access MosaicML's LLama 2 endpoint
# MAGIC - Deploy realtime model endpoint to perform RAG 
# MAGIC
# MAGIC Lakehouse AI is uniquely positioned to accelerate your Gen AI deployment.
# MAGIC
# MAGIC Interested in deploying your own models? Reach out to your account team!
