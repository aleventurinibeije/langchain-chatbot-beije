import os
import discord

from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from discord.ext import commands


load_dotenv()

template = """Sei Clara, BOT che si occuppa di illustrare il codice etico di Beije.
Context: {context}
Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template
)

if os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = DirectoryLoader("data/")
  index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(openai_api_key=os.environ.get('OPENAPI_API_KEY'), model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 2}),
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

chat_history = []

@bot.command()
async def question(ctx, *, question):
    global chat_history
    try:
        result = chain({"question": question, "chat_history": chat_history})
        answer = result['answer']
        print(answer)
        chat_history.append((question, answer))
        await ctx.send(answer)
    except Exception as e:
        print(f"Error occurred: {e}")
        await ctx.send("Mi spiace, non son riuscito a processare la domanda.")


bot.run(os.environ.get("DISCORD_TOKEN"))

