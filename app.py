from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

app = Flask(__name__)

# def get_subtitle(youtubelink):
#     video_id = youtubelink.split("=")[1]
#     transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#     for transcript in transcript_list:
#         Org_TS = transcript.fetch()
#         data = transcript.translate('en').fetch()
#     final_list = [item['text'] for item in data]
#     text = ' '.join(final_list)
#     return text

def get_subtitle(video_id):
    #video_id = youtubelink.split("=")[1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    for transcript in transcript_list:
        Org_TS = transcript.fetch()
        data = transcript.translate('en').fetch()
    final_list = [item['text'] for item in data]
    text = ' '.join(final_list)
    return text

# def generate_video_summary(video_transcript):
#     summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
#     chunks = [video_transcript[i:i + 1024] for i in range(0, len(video_transcript), 1024)]
#     summaries = []
#     for chunk in chunks:
#         summary = summarizer(chunk)
#         summaries.append(summary[0]['summary_text'])
#     final_summary = ' '.join(summaries)
#     return final_summary


#this endpoint fetches a transcript of a YouTube video, splits it into smaller text chunks, creates a vector 
# store from these chunks, and returns the original transcript as the response to the client.
@app.route('/summary', methods=['GET'])
def summary_api():
    url = request.args.get('url', '')
    video_id = url.split('=')[1]
    #summary = get_transcript(video_id)
    model = genai.GenerativeModel('gemini-pro')
    summary = get_subtitle(video_id)
    prompt = f"Summarize this video {summary}."
    response = model.generate_content(prompt)
    text_chunks = get_text_chunks(summary)
    get_vector_store(text_chunks)
    return response.text, 200

def get_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([d['text'] for d in transcript_list])
    return transcript

# def get_summary(video_transcript):
#     summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
#     chunks = [video_transcript[i:i + 1024] for i in range(0, len(video_transcript), 1024)]
#     summaries = []
#     for chunk in chunks:
#         summary = summarizer(chunk)
#         summaries.append(summary[0]['summary_text'])
#     final_summary = ' '.join(summaries)
#     return final_summary

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000) #chunk_size is the no. of characters in one chunk
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        youtubelink = request.form['youtubelink']
        raw_text = get_subtitle(youtubelink)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return jsonify({'message': 'Processing completed successfully.'}), 200
    return 'Flask Backend Running...'
ytblink = 'xyz'

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question', '')
    response = user_input(user_question)

    model = genai.GenerativeModel('gemini-pro')
    genai.configure(api_key="AIzaSyC_q1kel5pXMXbyZWzOtpMZPg8xEXp8fR8")

    prompt = f"""Provide web citation on: {response}. Provide multiple links in JSON format with object name 'citations' as
            [ citations 
            {{
                "url": "",
                "title": ""
            }}
            ]"""

    response1 = model.generate_content(prompt)
    result  = response1._result.candidates[0].content.parts[0].text
    cleaned_result = result.replace('```', '').replace('json', '')

    #return jsonify({'chatbot_response': response}), 200
    return jsonify({'chatbot_response': response, 'cleaned_result': cleaned_result}), 200

if __name__ == "__main__":
    app.run(debug=True)
