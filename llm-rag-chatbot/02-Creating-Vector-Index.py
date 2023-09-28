# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-nitin_wagh` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0912-181252-vaeks2ps/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 2/ Creating a Vector Search Index on top of our Delta Lake table
# MAGIC
# MAGIC <!-- <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-3.png?raw=true" style="float: right; width: 600px; margin-left: 10px"> -->
# MAGIC
# MAGIC We now have our knowledge base ready, and saved as a Delta Lake table within Unity Catalog (including permission, lineage, audit logs and all UC features).
# MAGIC
# MAGIC Typically, deploying a production-grade Vector Search index on top of your knowledge base is a difficult task. You need to maintain a process to capture table changes, index the model, provide a security layer, and all sorts of advanced search capabilities.
# MAGIC
# MAGIC Databricks Vector Search removes those painpoints.
# MAGIC
# MAGIC <!-- ## Databricks Vector Search
# MAGIC Databricks Vector Search is a new production-grade service that allows you to store a vector representation of your data, including metadata. It will automatically sync with the source Delta table and keep your index up-to-date without you needing to worry about underlying pipelines or clusters. 
# MAGIC
# MAGIC It makes embeddings highly accessible. You can query the index with a simple API to return the most similar vectors, and can optionally include filters or keyword-based queries.
# MAGIC
# MAGIC Vector Search is currently in Private Preview; you can [*Request Access Here*](https://docs.google.com/forms/d/e/1FAIpQLSeeIPs41t1Ripkv2YnQkLgDCIzc_P6htZuUWviaUirY5P5vlw/viewform)
# MAGIC
# MAGIC If you still do not have access to Databricks Vector Search, you can leverage [Chroma](https://docs.trychroma.com/getting-started) (open-source embedding database for building LLM apps). For an example end-to-end implementation with Chroma, pleaase see [this demo](https://www.dbdemos.ai/minisite/llm-dolly-chatbot/).  -->

# COMMAND ----------

# MAGIC %md
# MAGIC ## Document Embeddings 
# MAGIC
# MAGIC The first step is to create embeddings from the documents saved in our Delta Lake table. To do so, we need an LLM model specialized in taking a text of arbitrary length, and turning it into an embedding (vector of fixed size representing our document). 
# MAGIC
# MAGIC Embedding creation is done through LLMs, and many options are available: from public APIs to private models fine-tuned on your datasets.
# MAGIC
# MAGIC *Note: It is critical to ensure that the model is always the same for both embedding index creation and real-time similarity search. Remember that if your embedding model changes, you'll have to re-index your entire set of vectors, otherwise similarity search won't return relevant results.*

# COMMAND ----------

# DBTITLE 1,Install vector search package
# MAGIC %pip install pinecone-client
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=dbdemos $db=chatbot $reset_all_data=false

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating and registring our embedding model in UC
# MAGIC
# MAGIC Let's create an embedding model and save it in Unity Catalog. We'll then deploy it as serverless model serving endpoint. Vector Search will call this endpoint to create embeddings from our documents, and then index them.
# MAGIC
# MAGIC The model will also be used during realtime similarity search to convert the queries into vectors. This will be taken care of by Databricks Vector Search.
# MAGIC
# MAGIC #### Choosing an embeddings model
# MAGIC There are multiple choices for the embeddings model:
# MAGIC
# MAGIC * **SaaS API embeddings model**:
# MAGIC Starting simple with a SaaS API is a good option. If you want to avoid vendor dependency as a result of proprietary SaaS API solutions (e.g. Open AI), you can build with a SaaS API that is pointing to an OSS model. You can use the new [MosaicML Embedding](https://docs.mosaicml.com/en/latest/inference.html) endpoint: `/instructor-large/v1`. See more in [this blogpost](https://www.databricks.com/blog/using-ai-gateway-llama2-rag-apps)
# MAGIC * **Deploy an OSS embeddings model**: On Databricks, you can deploy a custom copy of any OSS embeddings model behind a production-grade Model Serving endpoint.
# MAGIC * **Fine-tune an embeddings model**: On Databricks, you can use AutoML to fine-tune an embeddings model to your data. This has shown to improve relevance of retrieval. AutoML is in Private Preview - [Request Access Here](https://docs.google.com/forms/d/1MZuSBMIEVd88EkFj1ehN3c6Zr1OZfjOSvwGu0FpwWgo/edit)
# MAGIC
# MAGIC Because we want to keep this demo simple, we'll directly leverage OpenAI, an external SaaS API. 
# MAGIC `mlflow.openai.log_model()` is a very convenient way to register a model calling OpenAI directly.
# MAGIC
# MAGIC
# MAGIC *Note: The Vector Search Private Preview might soon add support to directly plug an AI gateway for external SaaS API. This will make it even easier, without having to deploy the `mlflow.openai` model model. We will update this content accordingly.*
# MAGIC

# COMMAND ----------

import mlflow
#Let's setup our experiment
init_experiment_for_batch("llm-chatbot-rag", "embedding_gateway")

#Note that if you want to try the model, you need an openai key. We'll add it in the endpoint too as environement variable.
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("dbdemos", "openai")

#Enable Unity Catalog with mlflow registry
mlflow.set_registry_uri('databricks-uc')

client = MlflowClient()

try:
  #Get the model if it is already registered to avoid re-deploying the endpoint
  latest_model = client.get_model_version_by_alias(f"{catalog}.{db}.embedding_gateway", "prod")
  print(f"Our model is already deployed on UC: {catalog}.{db}.embedding_gateway")
except:
  #Simply create an mlflow openai flavor model
  with mlflow.start_run(run_name="embedding_open_ai") as run:
      mlflow.openai.log_model(model="text-embedding-ada-002", task="embeddings", artifact_path="embeddings")

  #Add the model to our catalog
  latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/embeddings', f"{catalog}.{db}.embedding_gateway")
  client.set_registered_model_alias(name=f"{catalog}.{db}.embedding_gateway", alias="prod", version=latest_model.version)

  #Make sure all other users can access the model for our demo(see _resource/00-init for details)
  set_model_permission(f"{catalog}.{db}.embedding_gateway", "ALL_PRIVILEGES", "account users")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Deploy our realtime model endpoint
# MAGIC
# MAGIC Note that the endpoint uses the `openai` MLFlow flavor, which requires the `OPENAI_API_KEY` environment variable to work. 
# MAGIC
# MAGIC Let's leverage Databricks Secrets to load the key when the endpoint starts. See the [documentation](https://docs.databricks.com/en/machine-learning/model-serving/store-env-variable-model-serving.html) for more details.

# COMMAND ----------

#Helper for the endpoint rest api, see details in _resources/00-init
serving_client = EndpointApiClient()
#Start the enpoint using the REST API (you can do it using the UI directly)
serving_client.create_enpoint_if_not_exists("dbdemos_embedding_endpoint", 
                                            model_name=f"{catalog}.{db}.embedding_gateway", 
                                            model_version = latest_model.version, 
                                            workload_size="Small", 
                                            environment_vars={"OPENAI_API_KEY": "{{secrets/dbdemos/openai}}"})

#Make sure all users can access our endpoint for this demo
#set_model_endpoint_permission("dbdemos_embedding_endpoint", "CAN_MANAGE", "users")

# COMMAND ----------

# MAGIC %md
# MAGIC Our endpoint is now deployed! You can directly [open it from the UI](/endpoints/dbdemos_embedding_endpoint) and visualize its performance!
# MAGIC
# MAGIC Let's run a REST query to try it in Python. As you can see, we send the `test sentence` doc and it returns an embedding representing our document.

# COMMAND ----------

# DBTITLE 1,Testing our Embedding endpoint
#Let's try to send some inference to our REST endpoint
dataset =  {"dataframe_split": {'data': ['test sentence']}}
import timeit

endpoint_url = f"{serving_client.base_url}/realtime-inference/dbdemos_embedding_endpoint/invocations"
print(f"Sending requests to {endpoint_url}")
starting_time = timeit.default_timer()
inferences = requests.post(endpoint_url, json=dataset, headers=serving_client.headers).json()
#print(f"Embedding inference, end 2 end :{round((timeit.default_timer() - starting_time)*1000)}ms {inferences}")
print(inferences['predictions'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating our Index
# MAGIC
# MAGIC As reminder, we want to add the index in the `databricks_documentation` table, indexing the column `content`. Let's review our `databricks_documentation` table:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM databricks_documentation

# COMMAND ----------

# MAGIC %sql 
# MAGIC ALTER TABLE databricks_documentation SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pinecone Index
# MAGIC
# MAGIC As reminder, we want to add the index in the `databricks_documentation` table, indexing the column `content`. Let's review our `databricks_documentation` table:

# COMMAND ----------

import pinecone      

pinecone.init(      
	api_key=dbutils.secrets.get("dbdemos", "pinecone_api_key"),      
	environment='gcp-starter'      
)      
dbdemos_index = pinecone.Index('dbdemos-index')

pinecone.describe_index("dbdemos-index")

# COMMAND ----------

table_name = "databricks_documentation"
database_name = "dbdemos.chatbot"
docs_df = spark.table(f"{database_name}.{table_name}")

# COMMAND ----------

endpoint_url = f"{serving_client.base_url}/realtime-inference/dbdemos_embedding_endpoint/invocations"
headers=serving_client.headers

# Create embeddings client
def create_embeddings(content):
  dataset =  {"dataframe_split": {'data': [content]}}
  #print(f"Sending requests to {endpoint_url}")
  inferences = requests.post(endpoint_url, json=dataset, headers=headers).json()
  return(inferences['predictions'][0])


create_embeddings("Create it ")
#Iterate over the DataFrame
for row in docs_df.rdd.collect():

    # Get the text to encode
    content = row.content
    id = row.id
    metadata = { "url": row.url, "content": row.content, "title": row.title }
    
    # Encode the text as a Pinecone vector embedding
    embedding = create_embeddings(row.content)
    #print(embedding)
    
    # Insert the vector embedding and metadata into the Pinecone index
    dbdemos_index.upsert(vectors=[{"id":str(id), "values":embedding, "metadata":metadata}])
    

# COMMAND ----------

# pinecone_df.write \
#     .option("pinecone.apiKey", dbutils.secrets.get("dbdemos", "pinecone_api_key")) \
#     .option("pinecone.environment", "gcp-starter") \
#     .option("pinecone.projectName", "dbdemos") \
#     .option("pinecone.indexName", "dbdemos-index") \
#     .format("io.pinecone.spark.pinecone.Pinecone") \
#     .mode("append") \
#     .save()

# COMMAND ----------

#Make sure you are able to query Pinecone Vector Search
question = "How can I track billing usage on my workspaces?"
embedding = create_embeddings(question)

results = dbdemos_index.query(
  vector=embedding,
  top_k=1,
  include_metadata=True
)

print(results['matches'][0]['metadata'])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Next step: Deploy our chatbot model with RAG
# MAGIC
# MAGIC We've seen how Databricks Lakehouse AI makes it easy to ingest and prepare your documents, and deploy a Vector Search index on top of it with just a few lines of code and configuration.
# MAGIC
# MAGIC This simplifies and accelerates your data projects so that you can focus on the next step: creating your realtime chatbot endpoint with well-crafted prompt augmentation.
# MAGIC
# MAGIC Open the [03-Deploy-RAG-Chatbot-Model]($./03-Deploy-RAG-Chatbot-Model) notebook to create and deploy a chatbot endpoint.
