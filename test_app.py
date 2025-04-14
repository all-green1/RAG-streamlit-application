import pytest
import os
import tempfile
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from app import HelpBot

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Set the global tmp_directory to our test directory
        import app
        app.tmp_directory = tmpdirname
        yield tmpdirname

@pytest.fixture
def test_db_path():
    """Create a temporary database path"""
    db_path = "test_chat_memory.db"
    # Remove existing test database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    return db_path

@pytest.fixture
def help_bot(temp_dir, monkeypatch):
    """Create a HelpBot instance with test configuration"""
    # Mock streamlit functions
    class MockStreamlit:
        def title(self, *args, **kwargs): pass
        def write(self, *args, **kwargs): pass
        def markdown(self, *args, **kwargs): pass
        def button(self, *args, **kwargs): return False
        def error(self, *args, **kwargs): pass
        def success(self, *args, **kwargs): pass
        def spinner(self, *args): return type('', (), {'__enter__': lambda x: None, '__exit__': lambda x,y,z,w: None})()

    mock_st = MockStreamlit()
    monkeypatch.setattr("streamlit.title", mock_st.title)
    monkeypatch.setattr("streamlit.write", mock_st.write)
    monkeypatch.setattr("streamlit.markdown", mock_st.markdown)
    monkeypatch.setattr("streamlit.error", mock_st.error)
    monkeypatch.setattr("streamlit.success", mock_st.success)
    monkeypatch.setattr("streamlit.spinner", mock_st.spinner)
    
    # Mock session state
    st.session_state = {}
    
    return HelpBot()

def test_database_setup(help_bot):
    """Test database initialization"""
    help_bot.setup_db()
    
    # Verify table creation
    conn = sqlite3.connect('chat_memory.db')  # Use the actual database name from your app
    cursor = conn.cursor()
    cursor.execute("""SELECT name FROM sqlite_master 
                     WHERE type='table' AND name='chat_history'""")
    table_exists = cursor.fetchone()
    conn.close()
    
    assert len(table_exists) > 0

def test_save_and_get_message(help_bot):
    """Test saving and retrieving messages"""
    # Setup
    session_id = "test-session"
    test_message = "Hello, test!"
    test_role = "user"
    
    # Save message
    help_bot.save_message_to_db(test_role, test_message, session_id)
    
    # Retrieve message
    history = help_bot.get_chat_history(session_id)
    
    assert len(history) > 1
    assert history[0][1] == session_id  # Check session_id
    assert history[0][2] == test_role   # Check role
    assert history[0][3] == test_message  # Check content

def test_document_processing(help_bot, temp_dir):
    """Test document loading and splitting"""
    # Create test document in the correct directory
    test_file = Path(temp_dir) / "test.txt"
    test_content = "This is a test document."
    test_file.write_text(test_content)
    
    # Test document loading
    documents = help_bot.load_docs()
    assert len(documents) > 0
    
    # Test document splitting
    splits = help_bot.split_docs(documents)
    assert len(splits) > 0

def test_chat_history_export(help_bot):
    """Test chat history export functionality"""
    # Add some test messages
    session_id = "test-session"
    messages = [
        ("user", "Hello"),
        ("assistant", "Hi there"),
        ("user", "How are you?")
    ]
    
    for role, content in messages:
        help_bot.save_message_to_db(role, content, session_id)
    
    # Get history and convert to DataFrame
    history = help_bot.get_chat_history()
    df = pd.DataFrame(history, columns=['timestamp', 'session_id', 'role', 'content'])
    
    
    assert all(col in df.columns for col in ['timestamp', 'session_id', 'role', 'content'])


def test_initialize_db_and_llm(help_bot, temp_dir, monkeypatch):
    """Test LLM and database initialization"""

    monkeypatch.setattr("langchain_community.embeddings.HuggingFaceEmbeddings", MagicMock())
    monkeypatch.setattr("langchain_chroma.Chroma", MagicMock())
    monkeypatch.setattr("langchain_openai.ChatOpenAI", MagicMock())


    # Mock load_docs to return a list of documents
    monkeypatch.setattr(help_bot, "load_docs", lambda: ["test document"])

    # Create test document
    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("This is a test document.")

    # Test initialization
    result = help_bot.initialize_db_and_llm()
    assert result is True
    assert help_bot.retriever is not None
    assert help_bot.llm is not None


def test_error_handling_no_documents(help_bot):
    """Test error handling when no documents are loaded"""
    result = help_bot.initialize_db_and_llm()
    assert result is False
    assert help_bot.retriever is None

from unittest.mock import MagicMock

def test_session_management(help_bot, monkeypatch):
    """Test session management"""
    
    # Mock streamlit session state
    mock_state = MagicMock()
    mock_state.session_id = None  # Mimic missing session_id initially

    # Patch st.session_state to use mock_state
    monkeypatch.setattr(st, "session_state", mock_state)

    session_id = help_bot.manage_sessions()
    
    assert session_id is not None
    assert len(session_id) > 0


@pytest.mark.parametrize("chunk_size,chunk_overlap", [
    (500, 20),
    (1000, 50),
    (200, 10)
])
def test_document_splitting_parameters(help_bot, temp_dir, chunk_size, chunk_overlap):
    """Test document splitting with different parameters"""
    # Create test document
    test_file = Path(temp_dir) / "test.txt"
    test_content = "This is a test document." * 100  # Make it longer
    test_file.write_text(test_content)
    
    documents = help_bot.load_docs()
    splits = help_bot.split_docs(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    assert len(splits) > 0
    # Check if most chunks are around the chunk_size
    for split in splits[:-1]:  # Exclude last chunk which might be smaller
        assert len(split.page_content) <= chunk_size + chunk_overlap