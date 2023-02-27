#用haystack的elasticsearch，针对本地的document

# from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers
# from haystack.nodes import FARMReader, TransformersReader
# from haystack.document_stores import ElasticsearchDocumentStore
# from haystack.nodes import BM25Retriever
#
# document_store = ElasticsearchDocumentStore(host="localhost", username="elastic", password="rTmD2wr3a_-qxvQq+QRL", index="document")
# doc_dir = "data/bookcorpus"
# # s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip"
# # fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
#
# docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
#
# document_store.write_documents(docs)
# retriever = BM25Retriever(document_store=document_store)
# candidate_documents = retriever.retrieve(
#     query="Father of Arya Stark is [masked]",
#     top_k=1,
# )


##接下来是直接从huggingface dataset放入elasticsearch
from datasets import load_dataset
import elasticsearch
data=load_dataset('wikipedia','20220301.en',cache_dir='D:\OneDrive\Desktop\corpus')
data=data['train']  #根据data具体是什么形态。Dataset.add_
es_client=elasticsearch.Elasticsearch('http://localhost:9200',http_auth=("elastic","rTmD2wr3a_-qxvQq+QRL"))
# data.add_elasticsearch_index(column='text',es_client=es_client,es_index_name="bookcorpus")
data.load_elasticsearch_index('text',es_client=es_client,es_index_name="wikipedia")
# query="Newton played as during Super Bowl 50 ."
num_samples=10
# scores,retrieved=data.get_nearest_examples("text",query,k=num_samples)
# print([" ".join(k) for k in [j.split() for j in [retrieved['text'][i] for i in range(num_samples)] if 'quarterback' not in j.lower()] if len(k)<2000][0])
query="Gregor Breinburg ( born [MAKSED]) ."
scores,retrieved=data.get_nearest_examples("text",query,k=num_samples)
print([" ".join(k) for k in [j.split() for j in [retrieved['text'][i] for i in range(num_samples)] if 'lahore' not in j.lower()] if len(k)<2000][0])