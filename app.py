import re

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from operator import itemgetter

import chainlit as cl
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()


document1 = PyMuPDFLoader(
    file_path="https://hiddenhistorycenter.org/wp-content/uploads/2016/10/PropagandaPersuasion2012.pdf"
).load()

document2 = PyMuPDFLoader(
    file_path="https://csmeyns.github.io/propaganda-everyday/pdf/odonnell-jowett-2018-what-is-propaganda.pdf"
).load()

document3 = PyMuPDFLoader(
    file_path="https://philpapers.org/archive/QUAP-2.pdf"
).load()

document4 = PyMuPDFLoader(
    file_path="https://www.ux1.eiu.edu/~bpoulter/2001/pdfs/propaganda.pdf"
).load()

document5 = PyMuPDFLoader(
    file_path="https://mediaeducationlab.com/sites/default/files/ms_vol10_br19_uvodnik_azurirana_verzija.pdf"
).load()


def metadata_generator(document, name):
    fixed_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", "!", "?"]
    )
    collection = fixed_text_splitter.split_documents(document)
    for doc in collection:
        doc.metadata["source"] = name
    return collection


documents = metadata_generator(document1, "Propaganda1") + metadata_generator(document2, "Propaganda2") + metadata_generator(document3, "Propaganda3") + metadata_generator(document4, "Propaganda4") + metadata_generator(document5, "Propaganda5")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Qdrant.from_documents(
    documents=documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="Propaganda"
)
alt_retriever = vectorstore.as_retriever()

## Generation LLM
llm = ChatOpenAI(model="gpt-4o")


RAG_PROMPT = """\
You are a propaganda expert. 
Given a provided context and question, you must answer if the question is propaganda or not.
The example of your response should be:
Whether the piece of text is propaganda or not.
If it is, say 🚨 PROPAGANDA!🚨 then cite the technique used and the relevant snippet of text where it is used.
If it is not, just answer "✅ Not Propaganda ✅"
Use real-time data to improve the quality of your answer and add better context
Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | alt_retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": rag_prompt | llm, "context": itemgetter("context")}
)
retrieval_augmented_qa_chain.invoke({"question": "This is a propaganda"})


@cl.on_message
async def handle_message(message):
    try:
        msg = cl.Message(content="")

        # Process the incoming question using the RAG chain
        async for event in retrieval_augmented_qa_chain.astream_events(
            input={"question": message.content},
            version="v2",
        ):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                await msg.stream_token(event.get("data").get("chunk").content)

        # Send the response back to the user
        await msg.send()

    except Exception as e:
        # Handle any exception and log it or send a response back to the user
        error_message = cl.Message(content=f"An error occurred: {str(e)}")
        await error_message.send()
        print(f"Error occurred: {e}")


# Run the ChainLit server
if __name__ == "__main__":
    try:
        cl.run()
    except Exception as e:
        print(f"Server error occurred: {e}")
