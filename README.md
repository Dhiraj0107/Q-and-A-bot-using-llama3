# Q-and-A-bot-using-llama3

This project aims to develop a Q&A bot designed to answer questions based on the article (P19-1496.pdf) that details creating a model capable of providing question-answering capabilities for genuine tweets used by journalists to write news articles. The project uses vector embeddings with Ollama embedding to store data in a Chroma vector store. The data is stored in chunks with some overlap between them. The model used to create the bot is Llama3 by Meta. Subsequently, RAG (Retrieval-Augmented Generation) is implemented using a prompt template that guides the model to answer questions accurately based on the provided context. Finally, the model and the prompt used to develop the chain, along with retrieval to interact with the vector store, generate the response.



## Below are screenshots of some questions answered by the model based on the pdf

![WhatsApp Image 2024-07-20 at 18 33 31_298c5970](https://github.com/user-attachments/assets/e5dfca81-1be5-4753-9f8f-bf9bf178adfc)

![WhatsApp Image 2024-07-20 at 18 45 34_7386b726](https://github.com/user-attachments/assets/6ae2a2a4-666d-4564-a282-869b647990bc)


## Below is the screenshot of the details captured in Langsmith

![Langsmith](https://github.com/user-attachments/assets/8241feee-79cc-4c5c-8e98-e9725c05996e)


#### References

https://github.com/krishnaik06/Updated-Langchain/blob/main/groq/llama3.py
