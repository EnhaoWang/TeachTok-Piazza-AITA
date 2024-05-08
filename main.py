from piazza_api import Piazza
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import time

def search_Pincone(question):
    load_dotenv()
    client = OpenAI() # api_key="OPENAI_API_KEY"
    pc = Pinecone()  
    index = pc.Index("ta")
    res = client.embeddings.create(input=[question], model="text-embedding-3-small")
    vector=res.data[0].embedding

    query_result={}
    for data_source in ['ppt_data', 'video_data', 'textbook_data', 'transcript_data']:
        top_k=10 if data_source=='textbook_data' else 1
        result=index.query(
        vector=vector,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
        namespace=data_source
        )      
        query_result[data_source]=[result_vector['metadata'] for result_vector in result['matches']]
        # query_result[data_source]=' '.join([result_vector['metadata']['text'] for result_vector in result['matches']])

    info_dict={}
    # text content can be from textbook or transcript
    #info_dict['text_content']=' '.join([result['text'] for result in query_result['textbook_data']])
    info_dict['text_content']=query_result['transcript_data'][0]['text']
    textbook_file=query_result['textbook_data'][0]['file_name']
    info_dict['textbook_tag']=f'<a href="http://infolab.stanford.edu/~ullman/mmds/{textbook_file}" target="_blank" rel="noreferrer">{textbook_file}</a>'

    ppt_file=query_result['ppt_data'][0]['file_name']
    ppt_page=query_result['ppt_data'][0]['page_num']
    info_dict['ppt_file']=ppt_file
    info_dict['ppt_tag']=f'<a href="http://www.mmds.org/mmds/v2.1/{ppt_file}" target="_blank" rel="noreferrer">{ppt_file}</a>'
    info_dict['ppt_page']=ppt_page
    info_dict['ppt_image_tag']=f'\n<img src = "https://raw.githubusercontent.com/EnhaoWang/TeachTok-Piazza-AITA/main/slides_image/{ppt_file}/page_{ppt_page}.png" width="768" height = "432" alt="image_file3 test">\n'

    video_id=query_result['video_data'][0]['video_id']
    info_dict['video_tag']=f'\n<iframe src = "https://www.youtube.com/embed/{video_id}" width="768" height = "432" alt="youtube link">\n'
    info_dict['video_url']=query_result['video_data'][0]['link']
    info_dict['timestamp']=query_result['video_data'][0]['time']
    
    return info_dict

def AI_component_syllabus(question):
    load_dotenv()
    template = """
    You are a component of an AI TA for a data mining course on Piazza. The student's question is: {question}
    Your task is to chech if the provided syllabus content can answer the question. If yes, answer it by yourself based on the syllabus content. If not, output 'no'(exact this word, without any other content).
    Below are the syllabus content for this course:

    Course information:
    The DSCI-553 course, titled "Foundations and Applications of Data Mining," is a graduate-level class at USC focusing on algorithms and techniques for analyzing large datasets, particularly using Spark. Here's a summarized overview of various aspects of the course:
    **Instructor and Contact Information**: The course is taught by Professor Wei-Min Shen. Students can contact him via email or his website for discussions outside of class.
    **Course Structure**: The course includes weekly quizzes and a combination of six homework assignments and a final project. The homework assignments require individual effort and utilize PySpark, with an optional bonus for also implementing them in Scala.
    **Prerequisites**: Students are expected to have a solid background in related courses like DSCI-551 and DSCI-552, and should be proficient in programming, algorithm design, and machine learning. Basic Unix skills are also recommended for handling programming assignments.
    **Grading**: The final grade is composed of quizzes (30%), homework (42%), a comprehensive exam (20%), and a data mining competition project (8%).
    **Materials and Resources**: The primary textbook is "Mining of Massive Datasets" by Rajaraman, Leskovec, and Ullman. Additional materials include lecture notes and possibly research papers. All course materials are available online.
    **Course Goals**: The course aims to equip students with practical skills in data mining to solve real-world problems. It involves case studies and applications to reinforce the theoretical knowledge taught.
    **Academic Policies**: The course adheres to strict academic integrity guidelines. Students are warned against plagiarism and other forms of academic dishonesty. The course also offers support for students with disabilities and promotes a safe learning environment free from discrimination and harassment.
    **Technology Requirements**: Students need to have access to a laptop capable of installing necessary software for coursework, mainly focusing on programming in Python and using Spark.

    Class schedule:
    1. **Week 1 (Starts January 8, 2024)**: Introduction to Data Mining, MapReduce.
    2. **Week 3 (Starts January 22, 2024)**: Frequent itemsets and Association rules. Homework 1 assigned.
    3. **Week 5 (Starts February 5, 2024)**: Recommendation Systems introduced. Homework 1 due and Homework 2 assigned.
    4. **Week 7 (Starts February 19, 2024)**: Analysis of Massive Graphs (Social Networks). Homework 2 due and Homework 3 assigned.
    5. **Week 10 (Starts March 11, 2024)**: Clustering massive data, Link Analysis. Homework 3 due, Homework 4 and Competition project assigned.
    6. **Week 12 (Starts March 25, 2024)**: Link Analysis continued. Homework 4 due and Homework 5 assigned.
    7. **Week 13 (Starts April 1, 2024)**: Web Advertising. Homework 5 due, Homework 6 assigned.
    8. **Week 14 (Starts April 8, 2024)**: Homework 6 due.
    9. **Week 15 (Starts April 15, 2024)**: Comprehensive Exam.
    10. **Week 16 (Starts April 22, 2024)**: Competition project due.
    """
    prompt = PromptTemplate(
        input_variables=['question'], template=template
    )

    llm = ChatOpenAI(model="gpt-4")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False)
    # generate reply by AI
    ai_response=llm_chain.predict(
        question=question)
    return ai_response

def AI_component_starter(chat_history, syllabus_response):
    load_dotenv()
    template = """
    You are a component of an AI TA for a data mining course on Piazza. Your rule is to receive student's question and analyse it.
    Below shows AI Response's chat history with the student, in which the first line is the subject of conversation and the current(unresponsed) question is in the last.

    {chat_history}

    Based on the above conversation history, first check if the question is related to data mining field. 
    If yes, output 'yes'(exact this word, without any other content).
    If no, check the content: 
    {syllabus_response}
    If the content answers the question, just return this content, exactly same.
    If still not, you can remind the student that the question is not related to data mining, and answer it by yourself based on the conversation history. 
    """
    prompt = PromptTemplate(
        input_variables=['chat_history', 'syllabus_response'], template=template
    )

    llm = ChatOpenAI(model="gpt-4")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False)
    # generate reply by AI
    ai_response=llm_chain.predict(
        chat_history=chat_history,
        syllabus_response=syllabus_response)
    return ai_response

def AI_component_transcript_textbook(chat_history, question, info_dict):
    load_dotenv()
    text_content=info_dict['text_content']
    textbook_tag=info_dict['textbook_tag']

    template = """
    You are a component of an AI TA for a data mining course on Piazza. Your rule is to generate a response with some provided information for the student's question.
    The question is: {question}.
    You should strictly follow the two steps to generate an response:
    1. Below is the some content:

    {text_content}

    First, you need to determine if the content is relevant to the question.
    If so, summerize the content above to generate a text answer. 
    If not, answer it on your own. You must output your answer only, don't say anything else, like 'content is not related to the question' or similar expressions.

    2. Based on the your response in the last step, write a catchy mnemonic(rhyme) for it, reminding student to better memorize the content.

    3. You should return the {textbook_tag}, the exact string(don't modify it, because this is an embedded file link in Piazza), and encourage student to read this textbook chapter.
    """
    prompt = PromptTemplate(
        input_variables=['question', 'text_content', 'textbook_tag'], template=template
    )

    llm = ChatOpenAI(model="gpt-4")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False)
    # generate reply by AI
    ai_response=llm_chain.predict(
        question=question,
        text_content=text_content,
        textbook_tag=textbook_tag)
    return ai_response

def AI_component_PPT(chat_history, question, info_dict):
    load_dotenv()
    ppt_file=info_dict['ppt_file']
    ppt_tag=info_dict['ppt_tag']
    ppt_page=info_dict['ppt_page']
    ppt_image_tag=info_dict['ppt_image_tag']

    template = """
    You are a component of an AI TA for a data mining course on Piazza. Your rule is to generate a response with some provided information for student's question.

    You should strictly follow the two steps to generate an response:

    1. You should return the {ppt_image_tag}, the exact string(don't modify it, because this is an embedded image in Piazza), and remind student to refer to file {ppt_file} on page {ppt_page}.

    2. You should indicate that the above PPT file is sourced from MMDs website and that the original slide file is {ppt_tag}, the exact string(don't modify it, because this is an embedded file link in Piazza).
    
    """
    prompt = PromptTemplate(
        input_variables=['ppt_file', 'ppt_tag', 'ppt_page', 'ppt_image_tag'], template=template
    )

    llm = ChatOpenAI(model="gpt-4")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False)
    # generate reply by AI
    ai_response=llm_chain.predict(
        ppt_file=ppt_file,
        ppt_tag=ppt_tag,
        ppt_page=ppt_page,
        ppt_image_tag=ppt_image_tag)
    return ai_response

def AI_component_video(chat_history, question, info_dict):
    load_dotenv()
    video_tag=info_dict['video_tag']
    video_url=info_dict['video_url']
    timestamp=info_dict['timestamp']

    template = """
    You are a component of an AI TA for a data mining course on Piazza. Your rule is to generate a response with some provided information for student's question.

    You should strictly follow the two steps to generate an response:

    1. You should return the {video_tag}, the exact string(don't modify it, because this is an embedded video in Piazza), and encourage the student to watch the lecture video at {timestamp}.

    2. In the next line, you should indicate that the original link of above video is {video_url}, the exact string(don't modify it, because this is a real YouTube video link).
    
    """
    prompt = PromptTemplate(
        input_variables=['video_tag', 'video_url', 'timestamp'], template=template
    )

    llm = ChatOpenAI(model="gpt-4")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False)
    # generate reply by AI
    ai_response=llm_chain.predict(
        video_tag=video_tag,
        video_url=video_url,
        timestamp=timestamp)
    return ai_response

def AI_component_summarizer(ai_responses):
    load_dotenv()
    textbook_response=ai_responses['textbook']
    ppt_response=ai_responses['ppt']
    video_response=ai_responses['video']

    template = """
    You are a component of an AI TA for a data mining course on Piazza. Your primary function is to collate and present the final responses with provided information. 

    Your task is: first, synthesize the content below.

    {textbook_response}
    {ppt_response}
    {video_response}

    Don't change the content in each sentence above. Your task is to add some line breaks between sentences to make the text visually appealing. Remove the item numbers at the beginning of some sentences, such as 1. 2. and so on.

    You must not modify any html tags like <img src=''>(with '<>' enclosing), because they are embedded elements in Piazza.

    Second, include an uplifting message to inspire students.
    
    """
    prompt = PromptTemplate(
        input_variables=['textbook_response', 'ppt_response', 'video_response'], template=template
    )

    llm = ChatOpenAI(model="gpt-4")
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False)
    # generate reply by AI
    ai_response=llm_chain.predict(
        textbook_response=textbook_response,
        ppt_response=ppt_response,
        video_response=video_response)
    return ai_response

def LLM_pipeline(chat_hist):
    chat_history=f'Conversation topic: {chat_hist[0][0]}\n'
    for conversation in chat_hist[1:-1]:
        chat_history+=f'Student: {conversation[0]}\nAI: {conversation[1]}\n'
    chat_history+=f'Student: {chat_hist[-1][0]}\nAI:\n'
    question=chat_hist[-1][0]
    syllabus_response=AI_component_syllabus(question)
    starter_reply=AI_component_starter(chat_history, syllabus_response)
    if starter_reply.lower()=='yes':
        info_dict=search_Pincone(question)
        ai_responses={}
        ai_responses['textbook']=AI_component_transcript_textbook(chat_history, question, info_dict)
        ai_responses['ppt']=AI_component_PPT(chat_history, question, info_dict)
        ai_responses['video']=AI_component_video(chat_history, question, info_dict)
        return AI_component_summarizer(ai_responses)
    else:
        return starter_reply

#login
p = Piazza()
p.user_login()
ds888 = p.network('luksbpak5t058g')
ai_id='lukt6e4ul1k5kj'

while 1:
    posts = list(ds888.iter_all_posts(sleep=0.5))
    question_posts=[]
    for post in posts:
        if post['type'] == 'question':
            question_posts.append(post)
    # Not considering the situation where there are two unanswered questions in the post simultaneously.
    for post in question_posts:
        chat_hist=[]
        post_id=post['id']
        # Take the latest content of the post (as the post may be edited), but the ID of the poster must be the oldest.
        stu=post['history'][-1]['uid']
        subject=post['history'][0]['subject']
        stu_question=post['history'][0]['content']  
        chat_hist.append([subject])
        chat_hist.append([stu_question])  
        if post['no_answer']==1: # Assume that the AI responds faster than any other instructor, so if there is a response, it must be from the AI.
            ai_response=LLM_pipeline(chat_hist)
            ds888.create_instructor_answer({'id': post_id}, content=ai_response, revision = 100, anonymous=False)
        else:
            unanswered_followup_id=''
            for child in post['children']:
                if child['type']=='followup' and child['uid']==stu:
                    replied=0
                    for reply in child['children']:
                        if reply['type']=='feedback' and reply['uid']==ai_id: # Follow-up only considers the first response from the AI (if the AI provides multiple responses, although this scenario is not likely to occur).
                            replied=1
                            chat_hist.append([child['subject'], reply['subject']])
                            break
                    if replied==0:
                        unanswered_followup_id=child['id']
                        chat_hist.append([child['subject']])
                        break
                elif child['type']=='i_answer': # and child['uid']==ai_id
                    chat_hist[1].append(child['history'][0]['content']) # Only consider the last response from the instructor in the post (as the instructor's responses may be edited, and the last one may not be from the AI).
            if unanswered_followup_id != '':
                ai_response=LLM_pipeline(chat_hist)
                ds888.create_reply({"id": unanswered_followup_id}, content=ai_response , anonymous=False)   

    time.sleep(10)