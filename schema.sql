-- Create courses table
CREATE TABLE IF NOT EXISTS courses (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    user_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create subjects table
CREATE TABLE IF NOT EXISTS subjects (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    course_id UUID REFERENCES courses(id),
    vector_db_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create questions table
CREATE TABLE IF NOT EXISTS questions (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    guideline_vector_db_url TEXT,
    guideline_text_box TEXT,
    is_guideline_textbox BOOLEAN DEFAULT FALSE,
    sample_answers_vector_db_url TEXT,
    sample_answers_textbox TEXT,
    is_sample_answer_textbox BOOLEAN DEFAULT FALSE,
    instructions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    subject_id UUID REFERENCES subjects(id)
); 