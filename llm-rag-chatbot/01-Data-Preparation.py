# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-nitin_wagh` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0912-181252-vaeks2ps/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ Data preparation for LLM Chatbot RAG
# MAGIC
# MAGIC ## Building our knowledge base and preparing our documents for Databricks Vector Search
# MAGIC
# MAGIC <img src="https://github.com/nitinbricks/databricks_pinecone_rag/blob/main/Databricks_Pinecone_RAG_architecture.png?raw=true" style="float: right; width: 800px; margin-left: 10px">
# MAGIC
# MAGIC In this notebook, we'll prepare data for our Vector Search Index.
# MAGIC
# MAGIC Preparing high quality data is key for your chatbot performance. We recommend taking time implementing this with your own dataset.
# MAGIC
# MAGIC For this example, we will use Databricks documentation from [docs.databricks.com](docs.databricks.com):
# MAGIC - Download the web pages
# MAGIC - Split the pages in small chunks
# MAGIC - Extract the text from the HTML content
# MAGIC
# MAGIC Thankfully, Lakehouse AI not only provides state of the art solutions to accelerate your AI and LLM projects, but also to accelerate data ingestion and preparation at scale.
# MAGIC
# MAGIC *Note: While some processing in this notebook is specific to our dataset (exmple: splitting chunks around `h2` elements), **we strongly recommend getting familiar with the overall process and replicate that on your own dataset**.*

# COMMAND ----------

# DBTITLE 1,Install required external libraries 
# installing a Python library to help us extract data out of HTML and XML files
# installing a tokenizer library 
%pip install beautifulsoup4==4.11.1 tiktoken==0.4.0

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=dbdemos $db=chatbot $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Extracting Databricks documentation sitemap and pages
# MAGIC
# MAGIC <!-- <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-1.png?raw=true" style="float: right; width: 600px; margin-left: 10px"> -->
# MAGIC
# MAGIC First, let's create our raw dataset as a Delta table.
# MAGIC
# MAGIC For this demo, we will directly download a few documentation pages from `docs.databricks.com`  and save the HTML content.
# MAGIC
# MAGIC Here are the main steps:
# MAGIC
# MAGIC - Run a quick script to extract the page URLs from the `sitemap.xml` file
# MAGIC - Download the web pages
# MAGIC - Use BeautifulSoup to extract the ArticleBody
# MAGIC - Save the result in a Delta Lake table
# MAGIC
# MAGIC *Note: for faster execution time, we will only download ~100 pages. Make sure you use these pages to ask questions to your model and see RAG in action!*

# COMMAND ----------

# DBTITLE 1,Extract the pages url from the sitemap.xml file
import requests
import xml.etree.ElementTree as ET

# Fetch the XML content from sitemap
response = requests.get("https://docs.databricks.com/en/doc-sitemap.xml")
root = ET.fromstring(response.content)
# Find all 'loc' elements (URLs) in the XML
urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
print(f"{len(urls)} Databricks documentation pages found")

#Let's take only the first 100 documentation pages to make this demo faster:
urls = urls[:100]

# COMMAND ----------

# DBTITLE 1,Download Databricks Documentation HTML pages as Raw Delta Lake table
import requests
import pandas as pd
import concurrent.futures
from bs4 import BeautifulSoup
import re

# Function to fetch HTML content for a given URL
def fetch_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# Function to process a URL and extract text from the specified div
def process_url(url):
    html_content = fetch_html(url)
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
        article_div = soup.find("div", itemprop="articleBody")
        if article_div:
            article_text = str(article_div)
            return {"url": url, "text": article_text.strip()}
    return None

# Use a ThreadPoolExecutor with 10 workers
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    results = list(executor.map(process_url, urls))

# Filter out None values (URLs that couldn't be fetched or didn't have the specified div)
valid_results = [result for result in results if result is not None]

#Save the content in a raw table
spark.createDataFrame(valid_results).write.mode('overwrite').saveAsTable('raw_documentation')
spark.sql("ALTER TABLE raw_documentation SET OWNER TO `account users`;")
display(spark.table('raw_documentation').limit(2))

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS databricks_documentation  (id BIGINT GENERATED BY DEFAULT AS IDENTITY, url STRING, content STRING, title STRING);
# MAGIC ALTER TABLE databricks_documentation SET OWNER TO `account users`;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Splitting documentation pages into small chunks
# MAGIC
# MAGIC <!-- <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-2.png?raw=true" style="float: right; width: 600px; margin-left: 10px"> -->
# MAGIC
# MAGIC LLM models typically have a maximum input window size, and you won't be able to compute embbeddings for very long texts.
# MAGIC In addition, the bigger your context is, the longer inference will take.
# MAGIC
# MAGIC Document preparation is key for your model to perform well, and multiple strategies exist depending on your dataset:
# MAGIC
# MAGIC - Split document in small chunks (paragraph, h2...)
# MAGIC - Truncate documents to a fixed length
# MAGIC - The chunk size depends of your content and how you'll be using it to craft your prompt. Adding multiple small doc chunks in your prompt might give different results than sending only a big one.
# MAGIC - Split into big chunks and ask a model to summarize each chunk as a one-off job, for faster live inference.
# MAGIC
# MAGIC ### LLM Window size and Tokenizer
# MAGIC
# MAGIC The same sentence might return different tokens for different models. LLMs are shipped with a `Tokenizer` that you can use to count how many tokens will be created for a given sequence (usually more than the number of words) (see [Hugging Face documentation](https://huggingface.co/docs/transformers/main/tokenizer_summary) or [OpenAI](https://github.com/openai/tiktoken))
# MAGIC
# MAGIC Make sure the tokenizer and context size limit you'll be using here matches your embedding model. To do so, we'll be using the `tiktoken` library to count GPT-3.5 tokens with its tokenizer: `tiktoken.encoding_for_model("gpt-3.5-turbo")`

# COMMAND ----------

# DBTITLE 1,Counting our tokens using tiktoken
import tiktoken
#Create our tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

#Truncate the given text to the number of token.
def truncate(text, tokenizer, max_tokens = 4000):
    token_count = len(tokenizer.encode(text))
    if token_count <= max_tokens:
        return text
    # Tokenize the text to get the tokens
    tokens = tokenizer.encode(text)
    # Truncate the tokens to the desired max_tokens
    truncated_tokens = tokens[:max_tokens]
    # Convert tokens back to text
    return tokenizer.decode(truncated_tokens)

#Let's try counting tokens  
sentence = "Hello, How are you? We are building an api for a chatbot."

token_count = len(tokenizer.encode(sentence))
print(f"This sentence has {token_count} tokens")
truncated_sentence = truncate(sentence, tokenizer, max_tokens = 10)
print(f"truncated to 10 tokens: {truncated_sentence}")  

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Splitting our big documentation page in smaller chunks (h2 sections)
# MAGIC
# MAGIC In this demo, we have a few big documentation articles, which are too big for our models. We'll split these articles between HTML `h2` tags, and ensure that each chunk isn't bigger than 4000 tokens.
# MAGIC
# MAGIC Let's also remove the HTML tags to send plain text to our model.

# COMMAND ----------

from bs4 import BeautifulSoup

# Remove multiple line breaks and truncate the model
def cleanup_and_truncate_text(text, tokenizer, max_tokens = 4000):
    return truncate(re.sub(r'\n{3,}', '\n\n', text).strip(), tokenizer, max_tokens)
    
#Split the text in chunk between 1000 and 4000 tokens
#This considers that our sections between H2 are of decent size (not > 4000 tokens), which is the case with our corpus. 
#H2 Sections longer than 4000 will be truncated.
def split_html_by_h2(soup, html_content, tokenizer, max_tokens = 4000, min_chunk_size = 1000):
    chunks = []
    last_index = 0
    for element in soup.find_all(['h2']):
        h2_position = html_content.find(str(element))
        chunk = html_content[last_index:h2_position]
        chunk_text = BeautifulSoup(chunk).get_text()
        # Split on the next H2 only if we have more than half the max. 
        # This prevents from having too small chunks
        if len(tokenizer.encode(chunk_text)) > max_tokens/2:
            last_index = h2_position
            chunks.append(cleanup_and_truncate_text(chunk_text, tokenizer, max_tokens))
    #Append the last chunk
    chunk = html_content[last_index:]
    chunk_text = BeautifulSoup(chunk).get_text()
    if len(tokenizer.encode(chunk_text)) > min_chunk_size:
        chunks.append(cleanup_and_truncate_text(chunk_text, tokenizer, max_tokens))
    return chunks
  
#Let's try to split our doc between h2:
doc = """<h1>This is a title</h1>
<h2>Subtitle 1</h2>
Some description 1
<h2>Subtitle 2</h2>
And description 2"""

#This text will be split in 2 parts, and the split is at an h2 element
for split in split_html_by_h2(BeautifulSoup(doc), doc, tokenizer, max_tokens=20, min_chunk_size=3):
  print(split)
  print("---------")

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's now split our entire dataset using this function using a pandas UDF.
# MAGIC
# MAGIC We will also extract the title from the page (based on the `h1` tag)

# COMMAND ----------


#Count number of tokens, if we exceed our limit will try to split it based on h2, or truncate it if no h2 exists.
def split_text(text, tokenizer, max_tokens = 4000, min_chunk_size = 100):
    soup = BeautifulSoup(text)
    txt = soup.get_text()
    token_count = len(tokenizer.encode(txt))
    if token_count > max_tokens:
        return split_html_by_h2(soup, text, tokenizer, max_tokens, min_chunk_size)
    else:
        return [re.sub(r'\n{3,}', '\n\n', txt).strip()]

#transform the html as text chunks. Will cut to 4000 tokens
@pandas_udf("array<string>")
def extract_chunks(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Load the model tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for serie in iterator:
        # get a summary for each row
        yield serie.apply(lambda x: split_text(x, tokenizer, max_tokens = 4000, min_chunk_size = 100))
        

@pandas_udf("string")
def clean_html(serie):
    return serie.apply(lambda x: BeautifulSoup(x).get_text())

df = (spark.table('raw_documentation')
        # Define the regular expression pattern (we could also have used soup)
        .withColumn("title", F.regexp_extract(col("text"), "<h1>(.*?)<\/h1>", 1))
        .withColumn("title", clean_html(col("title")))
        .withColumn('content', extract_chunks(col('text')))
        .drop('text'))
#Explode as we'll have multiple chunks per page
df = df.withColumn('content', F.explode('content'))
#Save back the results to our final table
#Note that we only do it if the table is empty, because it'll trigger an full indexation and we want to avoid this
if not spark.catalog.tableExists(f"{catalog}.{db}.databricks_documentation") or spark.table("databricks_documentation").count() < 50:
  df.write.mode('overwrite').saveAsTable('databricks_documentation')

# COMMAND ----------

display(spark.table("databricks_documentation"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Our dataset is now ready! Let's create our Vector Search Index.
# MAGIC
# MAGIC Our dataset is now ready, and saved as a Delta Lake table.
# MAGIC
# MAGIC We could easily deploy this part as a production-grade job, leveraging Delta Live Table capabilities to incrementally consume and cleanup document updates.
# MAGIC
# MAGIC Remember, this is the real power of the Lakehouse: one unified platform for data preparation, analysis and AI.
# MAGIC
# MAGIC Next: Open the [02-Creating-Vector-Index]($./02-Creating-Vector-Index) notebook and create our embedding endpoint and index.
