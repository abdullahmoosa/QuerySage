import streamlit as st
from datetime import datetime
import uuid
import psycopg2
from pinecone import Pinecone
from langchain.chains import RetrievalQA
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import tempfile
import logging
import json
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from docx import Document
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "courseassesmentsystem"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
logger.info("Initializing Pinecone connection")

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"Creating new Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine'
    )
    logger.info(f"Successfully created Pinecone index: {INDEX_NAME}")
index = pc.Index(INDEX_NAME)

# Initialize PostgreSQL connection
try:
    conn = psycopg2.connect(
        dbname="postgres", user="postgres", password="postgres", host="localhost"
    )
    cursor = conn.cursor()
    logger.info("Successfully connected to PostgreSQL database")
except Exception as e:
    logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
    raise

user_id = "0194b305-f06e-7661-810e-8f35c12ab058"


def process_pdf(file_content) -> str:
    """Process PDF file and return its text content"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        text_content = []
        
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())
            
        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def process_docx(file_content) -> str:
    """Process DOCX file and return its text content"""
    try:
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
                
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_content.append(" | ".join(row_text))
        
        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        raise

def process_txt(file_content) -> str:
    """Process TXT file and return its text content"""
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        # Try different encodings if UTF-8 fails
        try:
            return file_content.decode('latin-1')
        except Exception as e:
            logger.error(f"Error processing TXT: {str(e)}")
            raise

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and return its text content"""
    logger.info(f"Processing uploaded file: {uploaded_file.name}")
    
    try:
        file_content = uploaded_file.getvalue()
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.pdf':
            text_content = process_pdf(file_content)
        elif file_extension == '.docx':
            text_content = process_docx(file_content)
        elif file_extension == '.txt':
            text_content = process_txt(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        logger.info(f"Successfully processed file: {uploaded_file.name}")
        return text_content
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        raise


def split_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
    """Split text into smaller chunks"""
    logger.debug(f"Splitting text with chunk_size={chunk_size}, overlap={chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logger.debug(f"Split text into {len(chunks)} chunks")
    return chunks


def create_vector_store(documents: List[str], namespace: str) -> LangchainPinecone:
    """Create a Pinecone vector store with the specified namespace"""
    logger.info(f"Creating vector store in namespace: {namespace}")
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Create vector store
        vector_store = LangchainPinecone.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )
        logger.info(f"Successfully created vector store in namespace: {namespace}")
        
        total_chunks = 0
        # Process each document
        for doc_index, doc in enumerate(documents, 1):
            # Split the document into smaller chunks
            chunks = split_text(doc)
            total_chunks += len(chunks)
            
            logger.info(f"Processing document {doc_index}/{len(documents)} with {len(chunks)} chunks")
            
            # Add chunks in batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                try:
                    vector_store.add_texts(batch)
                    logger.info(f"Successfully processed batch {i//batch_size + 1} in document {doc_index}")
                    # Add a small progress indicator
                    st.write(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1} in document {doc_index}: {str(e)}")
                    continue
        
        logger.info(f"Completed vector store creation. Total chunks processed: {total_chunks}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        raise


def execute_db_query(query: str, params: tuple = None, fetch: bool = False) -> Any:
    """Execute a database query with logging"""
    try:
        logger.debug(f"Executing query: {query} with params: {params}")
        cursor.execute(query, params)
        
        if fetch:
            result = cursor.fetchall()
            logger.debug(f"Query returned {len(result)} rows")
            return result
        else:
            conn.commit()
            logger.debug("Query executed successfully")
            return None
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        conn.rollback()
        raise


def main():
    st.set_page_config(page_title="RAG Course Assessment System", layout="wide")

    # Initialize session state
    if "current_course" not in st.session_state:
        st.session_state.current_course = None
    if "current_subject" not in st.session_state:
        st.session_state.current_subject = None

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            [
                "Course Management",
                "Subject Management",
                "Question Management",
                "Answer Generation",
            ],
        )

    if page == "Course Management":
        show_course_management()
    elif page == "Subject Management":
        show_subject_management()
    elif page == "Question Management":
        show_question_management()
    else:
        show_answer_generation()


def show_course_management():
    st.title("Course Management")

    # Create new course
    with st.form(key="course_form"):
        st.subheader("Create New Course")
        course_name = st.text_input("Course Name")
        submit_course = st.form_submit_button("Create Course")

        if submit_course and course_name:
            course_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO courses (id, name, user_id) VALUES (%s, %s, %s)",
                (course_id, course_name, user_id),
            )
            conn.commit()
            st.success(f"Course '{course_name}' created successfully!")

    # Display existing courses
    st.subheader("Existing Courses")
    cursor.execute("SELECT * FROM courses WHERE user_id = %s", (user_id,))
    courses = cursor.fetchall()
    if courses:
        for course in courses:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"📚 {course[1]}")
            with col2:
                if st.button("Select", key=f"select_course_{course[0]}"):
                    st.session_state.current_course = course[0]
                    st.success(f"Selected course: {course[1]}")
    else:
        st.info("No courses created yet.")


def show_subject_management():
    st.title("Subject Management")

    if not st.session_state.current_course:
        st.warning("Please select a course first from Course Management.")
        return

    # Create new subject
    with st.form(key="subject_form"):
        st.subheader("Add New Subject")
        subject_name = st.text_input("Subject Name")
        guideline_files = st.file_uploader(
            "Upload Guideline Files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
        )
        submit_subject = st.form_submit_button("Add Subject")

        if submit_subject and subject_name:
            subject_id = str(uuid.uuid4())
            logger.info(f"Creating new subject: {subject_name} (ID: {subject_id})")
            
            # Process guideline files
            if guideline_files:
                documents = []
                for file in guideline_files:
                    try:
                        content = process_uploaded_file(file)
                        documents.append(content)
                        logger.info(f"Successfully processed guideline file: {file.name}")
                    except Exception as e:
                        logger.error(f"Failed to process file {file.name}: {str(e)}")
                        st.error(f"Error processing file {file.name}: {str(e)}")
                        continue
                
                if documents:
                    try:
                        # Create vector store with unique namespace for this subject
                        namespace = f"subject_{subject_id}"
                        vector_store = create_vector_store(documents, namespace)
                        vector_db_url = namespace

                        # Insert into database
                        execute_db_query(
                            "INSERT INTO subjects (id, name, course_id, vector_db_url) VALUES (%s, %s, %s, %s)",
                            (subject_id, subject_name, st.session_state.current_course, vector_db_url)
                        )
                        logger.info(f"Successfully created subject {subject_name} with ID {subject_id}")
                        st.success(f"Subject '{subject_name}' added successfully!")
                    except Exception as e:
                        logger.error(f"Failed to create subject: {str(e)}")
                        st.error(f"Failed to create subject: {str(e)}")
                else:
                    logger.warning("No files were successfully processed")
                    st.error("No files were successfully processed. Please try again.")

    # Display existing subjects
    st.subheader("Existing Subjects")
    subjects = execute_db_query(
        "SELECT * FROM subjects WHERE course_id = %s",
        (st.session_state.current_course,),
        fetch=True
    )
    logger.info(f"Retrieved {len(subjects)} subjects for course {st.session_state.current_course}")
    if subjects:
        for subject in subjects:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"📘 {subject[1]}")
            with col2:
                if st.button("Select", key=f"select_subject_{subject[0]}"):
                    st.session_state.current_subject = subject[0]
                    st.success(f"Selected subject: {subject[1]}")
    else:
        st.info("No subjects created yet.")


def show_question_management():
    st.title("Question Management")

    if not st.session_state.current_subject:
        st.warning("Please select a subject first from Subject Management.")
        return

    # Create new question
    with st.form(key="question_form"):
        st.subheader("Add New Question")
        question_text = st.text_area("Question Text")
        guideline_files = st.file_uploader(
            "Upload Guideline Files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
        )
        sample_answers_files = st.file_uploader(
            "Upload Sample Answers",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
        )
        instructions = st.text_area("Additional Instructions")
        submit_question = st.form_submit_button("Add Question")

        if submit_question and question_text:
            question_id = str(uuid.uuid4())
            guideline_vector_db_url = None
            
            # Process guideline files
            if guideline_files:
                documents = []
                for file in guideline_files:
                    try:
                        content = process_uploaded_file(file)
                        documents.append(content)
                    except Exception as e:
                        st.error(f"Error processing file {file.name}: {str(e)}")
                        continue
                
                if documents:  # Only proceed if we have successfully processed documents
                    # Create vector store with unique namespace for this question
                    namespace = f"question_{question_id}_guidelines"
                    vector_store = create_vector_store(documents, namespace)
                    guideline_vector_db_url = namespace

            # Process sample answers files
            sample_answers = []
            if sample_answers_files:
                for file in sample_answers_files:
                    try:
                        content = process_uploaded_file(file)
                        sample_answers.append(content)
                    except Exception as e:
                        st.error(f"Error processing file {file.name}: {str(e)}")
                        continue

            cursor.execute(
                "INSERT INTO questions (id, text, guideline_vector_db_url, sample_answers, instructions, subject_id) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    question_id,
                    question_text,
                    guideline_vector_db_url,
                    sample_answers,
                    instructions,
                    st.session_state.current_subject,
                ),
            )
            conn.commit()
            st.success("Question added successfully!")

    # Display existing questions
    st.subheader("Existing Questions")
    cursor.execute(
        "SELECT * FROM questions WHERE subject_id = %s",
        (st.session_state.current_subject,),
    )
    questions = cursor.fetchall()
    if questions:
        for question in questions:
            with st.expander(f"📝 {question[1][:100]}..."):
                st.write("Instructions:", question[4])
                st.write(f"Created: {question[5].strftime('%Y-%m-%d')}")
    else:
        st.info("No questions added yet.")


def show_answer_generation():
    st.title("Answer Generation")

    if not st.session_state.current_subject:
        st.warning("Please select a subject first from Subject Management.")
        return

    # Student information
    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("Student Name")
    with col2:
        student_id = st.text_input("Student ID")

    # Question selection
    logger.info(f"Fetching questions for subject: {st.session_state.current_subject}")
    
    # First, get subject information
    subject = execute_db_query(
        "SELECT * FROM subjects WHERE id = %s",
        (st.session_state.current_subject,),
        fetch=True
    )[0]
    logger.info(f"Retrieved subject information: {subject[1]} (ID: {subject[0]})")

    questions = execute_db_query(
        "SELECT * FROM questions WHERE subject_id = %s",
        (st.session_state.current_subject,),
        fetch=True
    )
    logger.info(f"Retrieved {len(questions)} questions from database")
    
    if questions:
        # Log question details for debugging
        for q in questions:
            logger.debug(f"Question ID: {q[0]}, Text: {q[1][:100]}...")

        selected_question = st.selectbox(
            "Select Question",
            options=[q[0] for q in questions],
            format_func=lambda x: next(q[1] for q in questions if q[0] == x)[:100] + "...",
        )

        # Generate answer button
        if st.button("Generate Answer"):
            if student_name and student_id:
                # Retrieve question details
                logger.info(f"Fetching details for question: {selected_question}")
                question = execute_db_query(
                    "SELECT * FROM questions WHERE id = %s",
                    (selected_question,),
                    fetch=True
                )[0]
                
                logger.info(f"Question details retrieved - ID: {question[0]}")
                logger.info(f"Question full details: Text: {question[1][:100]}..., Guidelines URL: {question[2]}")

                try:
                    # Initialize OpenAI embeddings
                    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
                    retrieved_content = []

                    # 1. First, get subject-level information (always available)
                    logger.info(f"Getting subject-level information from namespace: {subject[3]}")
                    subject_vector_store = LangchainPinecone.from_existing_index(
                        index_name=INDEX_NAME,
                        embedding=embeddings,
                        namespace=subject[3]  # subject vector_db_url
                    )
                    
                    # Get subject vector store stats
                    stats = index.describe_index_stats()
                    subject_vectors = stats.namespaces.get(subject[3], {}).get('vector_count', 0)
                    logger.info(f"Found {subject_vectors} vectors in subject information")
                    
                    # Retrieve relevant content from subject information
                    subject_retriever = subject_vector_store.as_retriever(search_kwargs={"k": 5})
                    subject_docs = subject_retriever.get_relevant_documents(question[1])  # Use question text directly
                    logger.info(f"Retrieved {len(subject_docs)} relevant chunks from subject information")
                    st.info(f"Using {len(subject_docs)} relevant chunks from subject information")
                    
                    # Add subject content to retrieved content
                    retrieved_content.extend([doc.page_content for doc in subject_docs])

                    # 2. Try to get question-specific guidelines if available
                    if question[2]:  # If there's a guideline vector DB URL
                        try:
                            logger.info(f"Getting question guidelines from namespace: {question[2]}")
                            question_vector_store = LangchainPinecone.from_existing_index(
                                index_name=INDEX_NAME,
                                embedding=embeddings,
                                namespace=question[2]
                            )
                            
                            # Get question guideline stats
                            question_vectors = stats.namespaces.get(question[2], {}).get('vector_count', 0)
                            if question_vectors > 0:
                                logger.info(f"Found {question_vectors} vectors in question guidelines")
                                
                                # Retrieve relevant content from guidelines
                                guideline_retriever = question_vector_store.as_retriever(search_kwargs={"k": 3})
                                guideline_docs = guideline_retriever.get_relevant_documents(question[1])
                                logger.info(f"Retrieved {len(guideline_docs)} relevant chunks from guidelines")
                                st.info(f"Using {len(guideline_docs)} relevant chunks from question guidelines")
                                
                                # Add guideline content to retrieved content
                                retrieved_content.extend([doc.page_content for doc in guideline_docs])
                        except Exception as e:
                            logger.warning(f"Could not use question guidelines: {str(e)}")
                    
                    # Get additional context if available
                    additional_context = ""
                    if question[3] and len(question[3]) > 0:  # Sample answers
                        logger.info("Adding sample answers to context")
                        additional_context += "\nSample Answers:\n" + "\n".join(question[3])
                    
                    if question[4]:  # Instructions
                        logger.info("Adding instructions to context")
                        additional_context += "\nInstructions:\n" + question[4]

                    # Prepare the prompt with all retrieved content
                    prompt = f"""Question: {question[1]}

Retrieved Information:
{'-' * 50}
{'\n'.join(retrieved_content)}
{'-' * 50}

{additional_context}

Please provide a comprehensive answer to the question using the retrieved information above."""

                    logger.info("Generating answer using combined information")
                    
                    # Generate answer using the combined information
                    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
                    answer = llm.predict(prompt)
                    
                    logger.info("Answer generated successfully")
                    logger.debug(f"Generated answer: {answer[:100]}...")
                    
                    st.success("Answer generated successfully!")
                    st.write(answer)

                    # Download options
                    st.subheader("Download Options")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download Answer Only",
                            data=answer,
                            file_name=f"answer_{student_id}.txt",
                        )
                    with col2:
                        st.download_button(
                            "Download Answer with Questions",
                            data=f"Question: {question[1]}\n\nAnswer: {answer}",
                            file_name=f"full_answer_{student_id}.txt",
                        )
                except Exception as e:
                    error_msg = f"Error during answer generation: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
            else:
                logger.warning("Attempted to generate answer without student information")
                st.warning("Please enter student name and ID.")
    else:
        logger.info("No questions found for the current subject")
        st.info("No questions available. Please add questions in Question Management.")


if __name__ == "__main__":
    main()
