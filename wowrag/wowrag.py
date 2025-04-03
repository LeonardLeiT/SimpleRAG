import os
from embedding import DocumentProcessor
from chat_chain import ChatRAGChain
from rag_model import MyEmbeddings
from llm_model import llm_deepseek
from tqdm import tqdm

def process_pdfs(pdf_dir='schwarz', persist_dir='database/chroma'):
    """
    Process all PDF files in the specified directory and create embeddings.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        persist_dir (str): Directory to store the vector database
    """
    # Initialize document processor
    doc_processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=100,
        persist_directory=persist_dir
    )
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    # Process each PDF file with progress bar
    pbar = tqdm(pdf_files, desc="Processing", unit="file", leave=True)
    for pdf_file in pbar:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pbar.set_description(f"Processing {pdf_file}")
        try:
            doc_processor.process_document(pdf_path)
        except Exception as e:
            print(f"\nError processing {pdf_file}: {str(e)}")
    
    print("\nDocument processing completed!")

def test_qa_system(persist_dir='database/chroma'):
    """
    Test the QA system with some example questions.
    
    Args:
        persist_dir (str): Directory containing the vector database
    """
    # Initialize the chat chain with RAG
    chat_chain = ChatRAGChain(
        persist_directory=persist_dir,
        llm_model=llm_deepseek(),
        embedding_model=MyEmbeddings()
    )
    
    # Test questions
    test_questions = [
        "What is the free energy criterion for schwarz crystal?",
        "What are the key properties of schwarz crystal?",
        "How does the Meissner effect work?",
        "What is the critical temperature in schwarz crystal?"
    ]
    
    print("\nTesting QA system with example questions:")
    print("-" * 50)
    
    # Test questions without progress bar
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("Answer: ", end="", flush=True)
        try:
            for chunk in chat_chain.chat(question, stream=True):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"Error getting answer: {str(e)}\n")

def main():
    """Main function to run the RAG system."""
    # Process PDFs and create embeddings
    print("Starting PDF processing...")
    process_pdfs()
    
    # Test the QA system
    print("\nStarting QA system testing...")
    test_qa_system()

if __name__ == "__main__":
    main()
