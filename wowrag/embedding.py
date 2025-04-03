from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
import os
import re
import uuid
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from rag_model import MyEmbeddings

class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        persist_directory: str = 'database/chroma',
        embedding_model: Optional[MyEmbeddings] = None
    ):
        """
        Initialize the DocumentProcessor with configuration parameters.
        
        Args:
            chunk_size (int): Size of text chunks for splitting
            chunk_overlap (int): Overlap between chunks
            persist_directory (str): Directory to persist vector store
            embedding_model: Custom embedding model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or MyEmbeddings()
        self.vectordb = None
        self.document_registry = {}  # Registry for storing document IDs and file paths

    def _generate_document_id(self, file_path: str) -> str:
        """
        Generate a simplified document ID based on file name and short UUID.
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            str: Simplified document ID
        """
        file_name = os.path.basename(file_path)
        short_id = str(uuid.uuid4())[:8]
        return f"{file_name}_{short_id}"

    def _add_document_to_registry(self, doc_id: str, file_path: str) -> None:
        """
        Add document to the registry.
        
        Args:
            doc_id (str): Document ID
            file_path (str): Path to the document
        """
        self.document_registry[doc_id] = file_path

    def get_document_info(self) -> dict:
        """
        Get information about all processed documents.
        
        Returns:
            dict: Dictionary containing document IDs and their file paths
        """
        return self.document_registry

    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text by removing unnecessary whitespace and irrelevant content.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split and clean line by line
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines and very short lines
            if len(line) < 10:
                continue
                
            # Skip pure number lines (likely page numbers)
            if line.isdigit():
                continue
                
            # Skip common header/footer markers
            if any(marker in line.lower() for marker in ['page', 'copyright', 'all rights reserved']):
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _load_document(self, file_path: str) -> List[Document]:
        """
        Load document based on file type (PDF or TXT).
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List[Document]: List of loaded documents
        """
        file_path = os.path.abspath(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyMuPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {file_path}")
        return documents

    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess documents by cleaning text content.
        
        Args:
            documents (List[Document]): List of documents to preprocess
            
        Returns:
            List[Document]: Preprocessed documents
        """
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)
        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks with improved splitting strategy.
        
        Args:
            documents (List[Document]): Documents to split
            
        Returns:
            List[Document]: Split documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False,
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
        return split_docs

    def _create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create and persist a vector store from documents.
        
        Args:
            documents (List[Document]): Documents to embed
            
        Returns:
            Chroma: Vector store
        """
        # Validate that all documents have required metadata
        for doc in documents:
            if not doc.metadata.get('document_id'):
                raise ValueError("Document missing document_id in metadata")
            if not doc.metadata.get('file_name'):
                raise ValueError("Document missing file_name in metadata")
            if not doc.metadata.get('file_path'):
                raise ValueError("Document missing file_path in metadata")
        
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        print(f"Created vector store with {vectordb._collection.count()} documents")
        return vectordb

    def process_document(self, file_path: str, metadata: dict = None) -> None:
        """
        Process a document and create a vector store.
        
        Args:
            file_path (str): Path to the document to process
            metadata (dict, optional): Additional metadata for the document
        """
        # Generate document ID
        doc_id = self._generate_document_id(file_path)
        self._add_document_to_registry(doc_id, file_path)
        
        # Load document
        documents = self._load_document(file_path)
        
        # Add document ID and metadata to each document
        for doc in documents:
            doc.metadata.update({
                "document_id": doc_id,
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                **(metadata or {})  # Add custom metadata if provided
            })
        
        # Preprocess documents
        documents = self._preprocess_documents(documents)
        
        # Split documents
        split_docs = self._split_documents(documents)
        
        # Validate that all documents have required metadata
        for doc in split_docs:
            if not doc.metadata.get('document_id'):
                raise ValueError("Document missing document_id in metadata")
            if not doc.metadata.get('file_name'):
                raise ValueError("Document missing file_name in metadata")
            if not doc.metadata.get('file_path'):
                raise ValueError("Document missing file_path in metadata")
        
        # Create vector store
        self.vectordb = self._create_vectorstore(split_docs)
        print(f"Processed document with ID: {doc_id}")
        print(f"Added {len(split_docs)} chunks from {file_path}")
        print(f"Total documents in store: {self.vectordb._collection.count()}")

    def add_document(self, file_path: str, metadata: dict = None) -> None:
        """
        Add a new document to the existing vector store with metadata.
        
        Args:
            file_path (str): Path to the new document to add
            metadata (dict, optional): Additional metadata for the document
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vectordb is None:
            raise ValueError("Vector store not initialized. Call process_document() first.")
            
        # Generate document ID
        doc_id = self._generate_document_id(file_path)
        self._add_document_to_registry(doc_id, file_path)
        
        # Load and process the new document
        documents = self._load_document(file_path)
        
        # Add document ID and metadata to each document
        for doc in documents:
            doc.metadata.update({
                "document_id": doc_id,
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                **(metadata or {})  # Add custom metadata if provided
            })
        
        documents = self._preprocess_documents(documents)
        split_docs = self._split_documents(documents)
        
        # Validate that all documents have required metadata
        for doc in split_docs:
            if not doc.metadata.get('document_id'):
                raise ValueError("Document missing document_id in metadata")
            if not doc.metadata.get('file_name'):
                raise ValueError("Document missing file_name in metadata")
            if not doc.metadata.get('file_path'):
                raise ValueError("Document missing file_path in metadata")
        
        # Add new documents to existing vector store
        self.vectordb.add_documents(split_docs)
        self.vectordb.persist()
        
        print(f"Added document with ID: {doc_id}")
        print(f"Added {len(split_docs)} new chunks from {file_path}")
        print(f"Total documents in store: {self.vectordb._collection.count()}")

    def search(self, query: str, k: int = 3, use_mmr: bool = True) -> List[Document]:
        """
        Search documents using similarity or MMR search with improved diversity.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            use_mmr (bool): Whether to use MMR search
            
        Returns:
            List[Document]: Retrieved documents
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vectordb is None:
            raise ValueError("Vector store not initialized. Call process_document() first.")
            
        if use_mmr:
            # Use MMR search with increased lambda parameter for better diversity
            results = self.vectordb.max_marginal_relevance_search(
                query, 
                k=k,
                fetch_k=20,  # Get more candidate results
                lambda_mult=0.7  # Increase diversity weight
            )
        else:
            # Use similarity search with duplicate filtering
            results = self.vectordb.similarity_search(query, k=k * 2)
            # Filter out duplicate content
            seen_content = set()
            unique_results = []
            for doc in results:
                content_hash = hash(doc.page_content[:100])  # Use first 100 characters as unique identifier
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(doc)
                if len(unique_results) >= k:
                    break
            results = unique_results
        
        # Validate metadata in search results
        for doc in results:
            # Ensure all required metadata fields exist
            doc.metadata.setdefault('document_id', doc.metadata.get('document_id', ''))
            doc.metadata.setdefault('file_name', doc.metadata.get('file_name', ''))
            doc.metadata.setdefault('file_path', doc.metadata.get('file_path', ''))
            
            # If any required field is empty, try to find it in the document registry
            if not doc.metadata['document_id'] or not doc.metadata['file_name'] or not doc.metadata['file_path']:
                # Try to find corresponding document info in registry
                for reg_doc_id, reg_file_path in self.document_registry.items():
                    if reg_file_path == doc.metadata.get('file_path', ''):
                        doc.metadata['document_id'] = reg_doc_id
                        doc.metadata['file_name'] = os.path.basename(reg_file_path)
                        doc.metadata['file_path'] = reg_file_path
                        break
        
        return results

    def get_vectorstore(self) -> Optional[Chroma]:
        """
        Get the current vector store.
        
        Returns:
            Optional[Chroma]: The current vector store or None if not initialized
        """
        return self.vectordb

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete specific documents from the vector store by their IDs.
        
        Args:
            document_ids (List[str]): List of document IDs to delete
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vectordb is None:
            raise ValueError("Vector store not initialized. Call process_document() first.")
            
        # Get the collection
        collection = self.vectordb._collection
        
        # Delete documents by their IDs
        collection.delete(ids=document_ids)
        print(f"Deleted {len(document_ids)} documents from the vector store")
        
        # Persist the changes
        self.vectordb.persist()

    def clear_database(self) -> None:
        """
        Clear all documents from the vector store and delete the persistence directory.
        
        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vectordb is None:
            raise ValueError("Vector store not initialized. Call process_document() first.")
            
        # Get the collection
        collection = self.vectordb._collection
        
        # Get all document IDs
        all_ids = collection.get()['ids']
        
        if all_ids:
            # Delete all documents
            collection.delete(ids=all_ids)
            print(f"Deleted all {len(all_ids)} documents from the vector store")
            
            # Persist the changes
            self.vectordb.persist()
            
            # Delete the persistence directory
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
                print(f"Deleted persistence directory: {self.persist_directory}")
        else:
            print("Vector store is already empty")

    def get_document_ids(self) -> List[str]:
        """
        Get all document IDs from the vector store.
        
        Returns:
            List[str]: List of document IDs
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vectordb is None:
            raise ValueError("Vector store not initialized. Call process_document() first.")
            
        return self.vectordb._collection.get()['ids']

# Example usage
if __name__ == "__main__":
    # Create a document processor instance with larger chunks
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=100,
        persist_directory='database/chroma'
    )
    
    # Process first PDF document
    pdf_path1 = "schwarz/Constrained minimal-interface structures in.pdf"
    print("\nProcessing first PDF:")
    print(f"File: {pdf_path1}")
    processor.process_document(pdf_path1)
    
    # Get document info for first PDF
    doc_registry = processor.get_document_info()
    print("\nFirst PDF Document Info:")
    for doc_id, file_path in doc_registry.items():
        print(f"Document ID: {doc_id}")
        print(f"File Path: {file_path}")
        print("--------------")
    
    # Add second PDF document
    pdf_path2 = "schwarz/Free_Energy_Criterion_for_SC_250320 (1).pdf"
    print("\nAdding second PDF:")
    print(f"File: {pdf_path2}")
    processor.add_document(pdf_path2)
    
    # Get updated document info
    print("\nUpdated Document Registry:")
    doc_registry = processor.get_document_info()
    for doc_id, file_path in doc_registry.items():
        print(f"Document ID: {doc_id}")
        print(f"File Path: {file_path}")
        print("--------------")
    
    # Get document counts
    print("\nDocument Statistics:")
    print(f"Total documents in registry: {len(doc_registry)}")
    print(f"Total chunks in vector store: {len(processor.get_document_ids())}")
    
    # Search in both documents with MMR
    question = "what is the free energy criterion for SC?"
    results = processor.search(question, k=3, use_mmr=True)
    
    # Print search results with document sources
    print("\nSearch Results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source Document: {doc.metadata['file_name']}")
        print(f"Document ID: {doc.metadata['document_id']}")
        print(f"File Path: {doc.metadata['file_path']}")
        print(f"Content Preview: {doc.page_content[:200]}...")
        print("--------------")
     