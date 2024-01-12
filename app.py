import os
import sys


from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import ChatOpenAI


os.environ["OPENAI_API_KEY"] = "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"

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

loader = DirectoryLoader("data/")
index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 2}),
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None

print(result['answer'])
