# End-to-end Medical Chatbot using Llama 2

## To set up and run this project locally, follow these steps:

1. Clone the Repository:
```bash
git clone https://github.com/negativenagesh/Medical_Chatbot-Llama2.git
```
2. Create a Conda Environment:
```bash
conda create --name mcbot python==3.8 -y
```
3. Activate the Environment:
```bash
conda activate mcbot
```
4. Install requirements
```bash
pip install -r requirements.txt
```

## What is ctransformers?

https://pypi.org/project/ctransformers/0.1.0/


CTransformers (C Transformers) is a library or framework for efficiently using transformer models in various applications. Let's break down the components and the use cases:

1. C Transformers:

* Transformers: A transformer is a type of deep learning model architecture that excels at handling sequential data and has become a cornerstone of natural language processing (NLP). It uses self-attention mechanisms to process and generate text efficiently.
* C: Typically, the "C" in CTransformers might stand for "compact", "concise", or "community-driven", indicating that the library focuses on making transformer models more accessible and efficient for a broader range of applications. However, without specific context, the exact meaning can vary.

2. Why its Used?

* Efficiency: CTransformers might offer optimized implementations of transformers that can run faster and require less computational power.
* Ease of Use: These libraries often provide user-friendly interfaces, pre-trained models, and utilities that make it easier for developers to integrate transformer models into their applications without needing deep expertise in machine learning.

3. Where It's Used?

* Natural Language Processing (NLP): Tasks like text summarization, translation, sentiment analysis, and question-answering.
* Computer Vision: Transformers are also being adapted for vision tasks, such as image classification and object detection.
* Multimodal Applications: Combining text, images, and other data types for tasks like video analysis and captioning.

4. Alternatives:

* Hugging Face Transformers: One of the most popular libraries for using transformer models. It provides a vast collection of pre-trained models and an easy-to-use interface.
* TensorFlow and Keras: TensorFlow offers its own implementations of transformers, which are widely used in various machine learning applications.
* PyTorch: PyTorch has its own transformer implementations, and the library is known for its flexibility and ease of experimentation.
* OpenAI GPT-3/4 APIs: Directly using APIs provided by OpenAI for tasks like text generation and summarization.
* AllenNLP: Another popular library for NLP tasks that includes implementations of transformer models.

5. Comparisons and why use ctransformers:

* Performance: If CTransformers offers better performance or lower latency compared to alternatives, it might be preferred for real-time applications.
* Resource Efficiency: If it requires less computational power or can run on more constrained hardware, it can be beneficial for deployment on edge devices or in resource-limited environments.

## What is sentence-transformers?

https://sbert.net 

The Sentence-Transformer model is a framework for embedding sentences into dense vector representations. It leverages architectures like BERT (Bidirectional Encoder Representations from Transformers) and its variants (e.g., RoBERTa, DistilBERT) to produce high-quality sentence embeddings that capture semantic information. The model is particularly useful for tasks requiring understanding the semantic similarity between sentences or text snippets.

1. Why It's Used?

* Semantic Similarity: The primary use case for sentence transformers is to compute the semantic similarity between sentences. This is crucial for tasks like duplicate question detection in forums, clustering similar documents, and retrieving semantically related text.

* Text Classification: By transforming sentences into embeddings, it becomes easier to apply various machine learning algorithms for classification tasks, such as sentiment analysis or topic classification.

* Information Retrieval: Sentence embeddings can significantly improve the performance of search engines by allowing more accurate matching of queries with relevant documents.

* Clustering: High-dimensional sentence embeddings can be used for clustering similar sentences or documents, which is valuable in organizing large datasets or identifying thematic patterns.

* Summarization: In text summarization tasks, sentence embeddings help in identifying and extracting the most relevant sentences that represent the core content.

2. Where its used?

* Question Answering Systems: To match user questions with relevant pre-existing answers or similar questions.

* Chatbots: Enhancing the ability of chatbots to understand user queries and provide relevant responses.

* Document Retrieval: Improving search results by retrieving documents based on semantic similarity rather than just keyword matching.

* Recommendation Systems: For recommending text-based content, such as news articles, research papers, or books, based on the user's interests.

* Paraphrase Identification: Detecting paraphrases in large text datasets, which is useful in data cleaning and deduplication tasks.

3. Alternatives:

* Universal Sentence Encoder (USE): Developed by    Google, USE provides similar functionality with different architecture optimizations. It is also designed to produce embeddings that can be used for various NLP tasks.

* InferSent: A model from Facebook AI that produces sentence embeddings using a combination of GloVe vectors and a BiLSTM network.

* QuickThoughts: Developed by researchers at Google, this model learns sentence representations by training on a sequence prediction task.

* GloVe and Word2Vec Averages: Averaging word embeddings from models like GloVe or Word2Vec can provide a simple, yet effective way to represent sentences.

* ELMo: Embeddings from Language Models (ELMo) generate contextualized word embeddings which can be averaged or otherwise combined to create sentence embeddings.

* Transformers Variants: Other transformer-based models, such as XLNet, T5, and GPT-3, can be fine-tuned to produce high-quality sentence embeddings.

## Pinecone Client:

https://pypi.org/project/pinecone-client/ 

Pinecone is a managed vector database service that is designed to handle high-dimensional vector data, which is commonly used in machine learning applications for tasks like similarity search and recommendation systems. The pinecone-client is the software library provided by Pinecone to interact with their service.

1. Why It's Used

* Vector Similarity Search:

Pinecone allows you to store, index, and query high-dimensional vectors efficiently. This is essential for applications that require finding similar items based on vector representations, such as recommendation systems and image similarity search.

* Scalability:

Pinecone is designed to handle large-scale vector data and can scale seamlessly as your data grows. This eliminates the need to manage and scale your own infrastructure.

* Performance:

Pinecone provides low-latency and high-throughput queries, which is critical for real-time applications like personalized recommendations or dynamic content retrieval.

* Ease of Use:

The pinecone-client library provides a simple and intuitive API for interacting with Pinecone's managed service, making it easy to integrate into existing applications and workflows.

2. Where It's Used

* Recommendation Systems:

E-commerce platforms can use Pinecone to recommend products to users based on the similarity of item vectors.

* Image and Video Search:

Platforms that need to find similar images or videos based on their visual content can use Pinecone for efficient similarity search.

* Natural Language Processing:

Applications that require semantic search or text similarity, such as chatbots or document retrieval systems, can benefit from Pinecone's vector search capabilities.

* Personalization:

Services that provide personalized content, such as news articles, music, or movies, can use Pinecone to deliver relevant content to users based on their preferences and behavior.

3. Alternatives

* Elasticsearch:

While primarily a text search engine, Elasticsearch has capabilities for vector similarity search through plugins and extensions. It is widely used and integrates well with various data sources.

* FAISS (Facebook AI Similarity Search):

FAISS is an open-source library developed by Facebook for efficient similarity search and clustering of dense vectors. It is highly optimized and performs well on large datasets.

* Annoy (Approximate Nearest Neighbors Oh Yeah):

Annoy is an open-source library developed by Spotify for approximate nearest neighbor search in high-dimensional spaces. It is easy to use and well-suited for read-heavy workloads.

* ScaNN (Scalable Nearest Neighbors):

Developed by Google, ScaNN is an open-source library for efficient similarity search in high-dimensional spaces. It offers a balance between accuracy and performance.

* Milvus:

Milvus is an open-source vector database designed for scalable similarity search. It supports various indexing methods and is optimized for large-scale vector data.

## LangChain

https://www.langchain.com/ 

LangChain is a library designed to facilitate the development of applications powered by language models, such as GPT-4. It provides a framework that simplifies the integration of various components needed for building complex language-based applications.

1. What is LangChain?

LangChain is a framework for developing applications using large language models (LLMs). It helps in chaining different components together, allowing developers to create complex workflows and pipelines that utilize the power of LLMs.

2. Why is LangChain used?

### LangChain is used for several reasons:

* Simplification: 

It abstracts away many of the complexities involved in working with language models directly.

* Modularity: 

Allows for the combination of various components like text generation, summarization, translation, etc., into a cohesive workflow.

* Flexibility: 

Supports the creation of custom pipelines and workflows tailored to specific use cases.

* Interoperability:

Easily integrates with other tools and libraries used in natural language processing (NLP).

3. Where is LangChain Used?

### LangChain can be used in a variety of applications, including but not limited to:

* Chatbots:

Building intelligent and context-aware conversational agents.

* Text Summarization:

Creating concise summaries of long documents.

* Content Generation:

Automating the creation of articles, blogs, and other content.

* Translation: 

Developing multilingual applications that require translation capabilities.

* Data Analysis: 

Using language models to extract insights and generate reports from data.

* Personal Assistants: 

Enhancing virtual assistants with advanced language understanding and generation capabilities.

4. Alternatives to LangChain

There are several alternatives to LangChain, each with its own set of features and use cases: Some of the popular ones include:

* Hugging Face Transformers:

A popular library for working with transformer models. It provides pre-trained models and tools for various NLP tasks. Use Cases: Text generation, translation, summarization, question answering, etc.

* spaCy:

An industrial-strength NLP library that provides tools for tokenization, part-of-speech tagging, named entity recognition, and more.
Use Cases: Text processing, named entity recognition, dependency parsing.

* NLTK (Natural Language Toolkit):

A library for working with human language data. It provides tools for text processing and classification. Use Cases: Educational purposes, text processing, linguistic research.

* OpenAI API:

Provides access to OpenAI's language models like GPT-3 and GPT-4 through an API. Use Cases: Text generation, conversation, content creation, etc.

* AllenNLP:

A library built on PyTorch for designing and evaluating deep learning models for NLP. Use Cases: Research and development in NLP, building custom models.

* TextBlob:

A simple library for processing textual data. It provides a simple API for diving into common NLP tasks. Use Cases: Text processing, sentiment analysis, classification.

## Libraries imported in [trials.ipynb](https://github.com/negativenagesh/Medical_Chatbot-Llama2/blob/main/trials.ipynb) 

1. LangChain (PromptTemplate, RetrievalQA, HuggingFaceEmbeddings):

* PromptTemplate:

This is used to create structured prompts for the language model, ensuring consistency and proper formatting in the inputs fed to the model. It helps in generating better and more reliable outputs.

* RetrievalQA:

This chain is designed for question-answering systems that need to retrieve relevant documents before generating an answer. It combines retrieval and generation in one seamless process.

* HuggingFaceEmbeddings:

These embeddings convert text into dense vector representations, capturing the semantic meaning. They are essential for tasks like similarity search, document retrieval, and clustering.

2. Pinecone:

* Pinecone is a vector database optimized for storing and querying high-dimensional vectors. It allows for fast similarity searches, making it ideal for applications where you need to find the most relevant documents or data points based on their embeddings.

3. PyMuPDFLoader and DirectoryLoader

* PyMuPDFLoader:

This loader is used for extracting text from PDF documents using the PyMuPDF library. It is useful when you need to process and analyze text content from PDFs.

* DirectoryLoader: 

This loader allows you to read multiple documents from a directory. It supports various file types and is helpful for batch processing and loading large sets of documents for analysis or indexing.

4. RecursiveCharacterTextSplitter

This tool splits long texts into smaller, manageable chunks based on character count. It ensures that the text fits within the token limits of language models and helps in efficient text processing and retrieval.

5. CTransformers

* This module is part of the LangChain library and provides an interface to use transformer models more efficiently. It can be particularly useful for deploying transformer models in a production environment where performance and resource optimization are critical.

6. PineconeHybridSearchRetriever

* The PineconeHybridSearchRetriever is a class in the LangChain library designed to perform hybrid search retrieval using Pinecone's vector database. This class combines both semantic (vector-based) and keyword (term-based) search techniques to retrieve the most relevant documents for a given query.

7. LangChainPinecone

* The Pinecone class from langchain.vectorstores, referred to as LangChainPinecone, is a wrapper around Pinecone's vector database service, designed to integrate seamlessly with the LangChain library. This class provides functionalities for storing, indexing, and searching vector embeddings, which are crucial for various natural language processing tasks such as semantic search, document retrieval, and more.

8. ContextualCompressionRetriever

* The ContextualCompressionRetriever from the langchain.retrievers module is a specialized class designed to enhance the efficiency and relevance of information retrieval in natural language processing tasks. This retriever compresses the context of documents or text data to prioritize the most relevant information, which is particularly useful for large datasets where processing and retrieving information can be resource-intensive.

9. LLMChainExtractor

* The LLMChainExtractor from the langchain.retrievers document_compressors module is a specialized component designed to extract and compress relevant information from documents using a large language model (LLM). This extractor is particularly useful in contexts where documents contain a lot of information, but only specific parts are relevant to the query.

10. from pinecone import Pinecone

* The Pinecone class from the pinecone module is the main interface for interacting with the Pinecone vector database service. Pinecone provides a highly scalable, low-latency infrastructure for managing and querying high-dimensional vector embeddings, which are crucial for a variety of machine learning and natural language processing tasks, such as semantic search, recommendation systems, and more.

# Creating Pinecone Index in [trials.ipynb](https://github.com/negativenagesh/Medical_Chatbot-Llama2/blob/main/trials.ipynb)

Creating a Pinecone index is essential for managing vector embeddings efficiently. In the context of chatbots, especially those requiring sophisticated natural language understanding and response generation, Pinecone provides the following benefits:

1. Efficient Vector Storage and Retrieval:

* Embedding Management: 

Chatbots often use embeddings (vector representations) of text data to understand and generate responses. Pinecone indexes store these embeddings efficiently, allowing for fast and scalable retrieval.

* Similarity Search: 

Pinecone allows for similarity searches, such as finding the closest embeddings to a given query. This is crucial for tasks like finding the most relevant previous conversation snippet or retrieving relevant knowledge base entries.

2. Scalability:

* Handling Large Datasets: 

As the amount of data grows, managing embeddings and ensuring quick retrieval becomes challenging. Pinecone is designed to handle large-scale vector data, making it ideal for chatbots with extensive conversation histories or large knowledge bases.

* Serverless Architecture: 

Pinecone’s serverless approach means it can scale dynamically based on the load, ensuring that your chatbot can handle varying levels of traffic without manual intervention.

3. Performance:

* Low Latency: 

Pinecone is optimized for low-latency operations, which is critical for chatbots that need to respond in real-time.

* High Throughput: 

It supports high-throughput operations, enabling the chatbot to handle multiple simultaneous requests efficiently.

## Usage in Chatbots:

Chatbots use vector embeddings to represent text data in a way that captures semantic meaning. Here’s how Pinecone fits into the architecture:

1. Intent Recognition:

* Embedding User Queries: 

When a user sends a message, the chatbot converts the text into an embedding using a pre-trained model (e.g., BERT, GPT).

* Similarity Search: 

The embedding is then used to search the Pinecone index to find the most similar previous queries, intents, or responses.

2. Contextual Understanding:

* Conversation History: 

The chatbot can store and retrieve embeddings of past conversation snippets to maintain context, ensuring more coherent and contextually relevant responses.

3. Knowledge Base Retrieval:

* FAQ Matching: 

For chatbots designed to answer FAQs, user queries can be converted into embeddings and matched against a pre-indexed knowledge base in Pinecone to find the most relevant answers.

# All Functions in [trials.ipynb](https://github.com/negativenagesh/Medical_Chatbot-Llama2/blob/main/trials.ipynb) briefly

1. def load_pdf(data)

* The load_pdf function is designed to load PDF files from a specified directory.

2. def text_split(extracted_data)  

* The text_split function splits a list of documents into smaller chunks. It uses RecursiveCharacterTextSplitter to divide the text into chunks of 500 characters with an overlap of 20 characters between chunks, and then returns these chunks.

3. def download_hugging_face_embeddings()

* The download_hugging_face_embeddings function initializes and returns a Hugging Face embeddings object using the sentence-transformers/all-MiniLM-L6-v2 model.

4. 

