from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from langchain import HuggingFaceHub
from langchain.schema import HumanMessage, AIMessage
from transformers import AutoTokenizer

template = """
### If you are a doctor, please answer the medical questions based on the patient's description
{chat_history}
### Input: {question}
### Output:
"""

model_id = "Vedant210604/doc_chat_tinyllama"
conv_model = HuggingFaceHub(
    huggingfacehub_api_token="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",
    repo_id=model_id,
    model_kwargs={'temperature': 0.3, 'max_new_tokens': 200, 'top_p':0.7,'num_return_sequence':1,'repetition_penalty': 1.2,'penalty_alpha': 0.6,})

@cl.on_chat_start
def quey_llm():
    prompt = PromptTemplate(input_variables=['chat_history', 'question'], template=template)
    memory = ConversationBufferMemory(memory_key='chat_history',max_token_limit=500,return_message = True)
    conv_chain = LLMChain(
        llm=conv_model,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    cl.user_session.set("llm_chain", conv_chain)

@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")

    response = await llm_chain.acall(message.content,
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])

    output_start = "### Output:\n"
    output = response["text"].split(output_start)[-1].strip()

    await cl.Message(output).send()
