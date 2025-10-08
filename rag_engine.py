from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
import os
from dotenv import load_dotenv
from pathlib import Path
import pickle

load_dotenv()

class EnhancedPharaonicRAG:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
                 chunk_size=500, chunk_overlap=150, parent_chunk_size=2000):
        """
        Initialize Enhanced RAG with:
        - Smart overlapping chunks
        - Parent-Child retrieval strategy
        - Context expansion
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        
        self.embeddings = None
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.qa_chain = None
        
        # Store both child chunks (for retrieval) and parent chunks (for context)
        self.child_documents = []
        self.parent_documents = []
        self.chunk_to_parent_map = {}  # Maps child chunk ID to parent chunk
        
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2000
        )
        
    def load_text_file(self, file_path):
        """Load text from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                return text
            except Exception as e:
                raise Exception(f"Error reading file: {str(e)}")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")
    
    def build_knowledge_base(self, file_path):
        """Build knowledge base with Parent-Child chunking strategy."""
        print(f"📚 Loading text from {file_path}...")
        text = self.load_text_file(file_path)
        
        print(f"✂️  Creating Parent-Child chunks with smart overlapping...")
        
        # Step 1: Create large parent chunks (for context)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=400,  # Large overlap for parents
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "، ", "؛ ", " "],
            keep_separator=True
        )
        
        parent_docs = parent_splitter.create_documents([text])
        print(f"   Created {len(parent_docs)} parent chunks (context)")
        
        # Step 2: Create smaller child chunks (for precise retrieval)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,  # Good overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "، ", "؛ ", " "],
            keep_separator=True
        )
        
        # Split each parent into children and maintain the relationship
        for parent_idx, parent_doc in enumerate(parent_docs):
            children = child_splitter.split_documents([parent_doc])
            
            for child_idx, child in enumerate(children):
                # Add metadata to track parent-child relationship
                child.metadata['parent_idx'] = parent_idx
                child.metadata['child_idx'] = child_idx
                child.metadata['chunk_id'] = f"p{parent_idx}_c{child_idx}"
                
                self.child_documents.append(child)
                self.chunk_to_parent_map[f"p{parent_idx}_c{child_idx}"] = parent_doc
        
        self.parent_documents = parent_docs
        
        print(f"   Created {len(self.child_documents)} child chunks (retrieval)")
        print(f"   Average overlap: {self.chunk_overlap} chars ({(self.chunk_overlap/self.chunk_size)*100:.1f}%)")
        
        # Show sample
        print(f"\n📄 Sample Child Chunk:")
        print(self.child_documents[0].page_content[:300])
        print("\n📄 Its Parent Context:")
        print(self.chunk_to_parent_map[self.child_documents[0].metadata['chunk_id']].page_content[:300])
        print("...\n")
        
        print(f"🧠 Setting up Enhanced Hybrid Retrieval...")
        
        # 1. BM25 Retriever on child chunks
        self.bm25_retriever = BM25Retriever.from_documents(self.child_documents)
        self.bm25_retriever.k = 6
        
        # 2. Semantic Retriever on child chunks
        print(f"   Generating embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = FAISS.from_documents(self.child_documents, self.embeddings)
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        
        # 3. Ensemble Retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6]  # Favor semantic slightly
        )
        
        print(f"🤖 Setting up Enhanced QA chain...")
        self._setup_qa_chain()
        
        print(f"✅ Enhanced knowledge base ready!")
        print(f"   - {len(self.parent_documents)} parent contexts")
        print(f"   - {len(self.child_documents)} searchable chunks")
        return self
    
    def _expand_context(self, retrieved_docs, num_neighbors=1):
        """
        Expand retrieved chunks by including neighboring chunks.
        This provides better context continuity.
        """
        expanded_docs = []
        seen_chunks = set()
        
        for doc in retrieved_docs:
            parent_idx = doc.metadata.get('parent_idx')
            child_idx = doc.metadata.get('child_idx')
            chunk_id = doc.metadata.get('chunk_id')
            
            if chunk_id in seen_chunks:
                continue
            
            # Get neighboring chunks from the same parent
            neighbors = []
            for offset in range(-num_neighbors, num_neighbors + 1):
                neighbor_id = f"p{parent_idx}_c{child_idx + offset}"
                
                # Find the neighbor chunk
                for child_doc in self.child_documents:
                    if child_doc.metadata.get('chunk_id') == neighbor_id:
                        if neighbor_id not in seen_chunks:
                            neighbors.append(child_doc)
                            seen_chunks.add(neighbor_id)
                        break
            
            # If neighbors found, add them; otherwise use parent
            if neighbors:
                expanded_docs.extend(neighbors)
            else:
                # Fallback to parent chunk for maximum context
                parent_doc = self.chunk_to_parent_map.get(chunk_id)
                if parent_doc:
                    expanded_doc = Document(
                        page_content=parent_doc.page_content,
                        metadata={'source': 'parent_context', **doc.metadata}
                    )
                    expanded_docs.append(expanded_doc)
        
        return expanded_docs[:8]  # Return top 8 expanded chunks
    
    def _setup_qa_chain(self):
        """Setup QA chain with context-aware retrieval."""
        template = """أنت مساعد خبير في التاريخ المصري القديم. لديك سياق موسع يشمل معلومات مترابطة.

⚠️ قواعد الإجابة:
1. استخدم المعلومات من السياق الموسع أدناه
2. السياق يحتوي على أجزاء مترابطة - استخدمها معاً لفهم أفضل
3. إذا وجدت المعلومة، أجب بشكل مفصل وشامل
4. إذا لم تجد المعلومة في السياق، قل: "هذه المعلومة غير موجودة في قاعدة البيانات الحالية"
5. اربط المعلومات من أجزاء مختلفة في السياق لإعطاء إجابة كاملة

السياق الموسع (مع أجزاء مترابطة):
==================
{context}
==================

السؤال: {question}

الإجابة الشاملة:"""

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Custom retriever that expands context - inherits from BaseRetriever
        class ContextExpandedRetriever(BaseRetriever):
            base_retriever: object
            expand_fn: object
            
            class Config:
                arbitrary_types_allowed = True
            
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
            ) -> List[Document]:
                """Get documents relevant to a query."""
                base_docs = self.base_retriever.get_relevant_documents(query)
                return self.expand_fn(base_docs, num_neighbors=1)
        
        expanded_retriever = ContextExpandedRetriever(
            base_retriever=self.ensemble_retriever,
            expand_fn=self._expand_context
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=expanded_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def save_knowledge_base(self, save_dir="enhanced_knowledge_base"):
        """Save the enhanced knowledge base."""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Save FAISS index
        self.vectorstore.save_local(f"{save_dir}/faiss_index")
        
        # Save all documents and mappings
        with open(f"{save_dir}/documents.pkl", 'wb') as f:
            pickle.dump({
                'child_documents': self.child_documents,
                'parent_documents': self.parent_documents,
                'chunk_to_parent_map': self.chunk_to_parent_map
            }, f)
        
        # Save config
        with open(f"{save_dir}/config.pkl", 'wb') as f:
            pickle.dump({
                'model_name': self.model_name,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'parent_chunk_size': self.parent_chunk_size
            }, f)
        
        print(f"💾 Enhanced knowledge base saved to {save_dir}/")
    
    def load_knowledge_base(self, save_dir="enhanced_knowledge_base"):
        """Load the enhanced knowledge base."""
        # Load config
        with open(f"{save_dir}/config.pkl", 'rb') as f:
            config = pickle.load(f)
            self.model_name = config['model_name']
            self.chunk_size = config['chunk_size']
            self.chunk_overlap = config['chunk_overlap']
            self.parent_chunk_size = config['parent_chunk_size']
        
        # Load documents
        with open(f"{save_dir}/documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.child_documents = data['child_documents']
            self.parent_documents = data['parent_documents']
            self.chunk_to_parent_map = data['chunk_to_parent_map']
        
        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load FAISS
        self.vectorstore = FAISS.load_local(
            f"{save_dir}/faiss_index",
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Recreate retrievers
        self.bm25_retriever = BM25Retriever.from_documents(self.child_documents)
        self.bm25_retriever.k = 6
        
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6]
        )
        
        self._setup_qa_chain()
        
        print(f"✅ Enhanced knowledge base loaded!")
        return self
    
    def ask(self, query, return_sources=False):
        """Ask a question with enhanced context retrieval."""
        if self.qa_chain is None:
            raise Exception("Knowledge base not initialized.")
        
        print(f"\n🔍 Searching with Enhanced Hybrid Retrieval + Context Expansion...")
        
        result = self.qa_chain.invoke({"query": query})
        
        response = {
            'answer': result['result'],
            'source_documents': result['source_documents']
        }
        
        if return_sources:
            return response
        else:
            return response['answer']
    
    def test_retrieval(self, query):
        """Test the enhanced retrieval system."""
        print(f"\n🔍 Testing Enhanced Retrieval: '{query}'\n")
        print(f"Database: {len(self.parent_documents)} parents, {len(self.child_documents)} children\n")
        
        # Get base retrieval
        base_docs = self.ensemble_retriever.get_relevant_documents(query)
        print("=" * 60)
        print("Base Retrieved Chunks:")
        print("=" * 60)
        for i, doc in enumerate(base_docs[:5], 1):
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            print(f"\n[Chunk {i}] ID: {chunk_id}")
            print(doc.page_content[:300])
            print("...")
        
        # Get expanded context
        expanded_docs = self._expand_context(base_docs, num_neighbors=1)
        print("\n" + "=" * 60)
        print("After Context Expansion (with neighbors):")
        print("=" * 60)
        for i, doc in enumerate(expanded_docs, 1):
            chunk_id = doc.metadata.get('chunk_id', 'parent')
            source = doc.metadata.get('source', 'child')
            print(f"\n[Expanded {i}] ID: {chunk_id} | Source: {source}")
            print(doc.page_content[:300])
            print("...")
        
        return expanded_docs
    
    def chat(self):
        """Interactive chat with enhanced retrieval."""
        print("\n" + "="*60)
        print("🏛️  Enhanced Pharaonic RAG - نظام محسّن")
        print("="*60)
        print("✨ Features:")
        print("   - Smart overlapping chunks (30% overlap)")
        print("   - Parent-Child retrieval strategy")
        print("   - Automatic context expansion")
        print("   - Hybrid search (BM25 + Semantic)")
        print("\nاسألني أي شيء عن مصر القديمة!")
        print("Type 'quit' or 'خروج' to exit.\n")
        
        while True:
            try:
                query = input("أنت: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q', 'خروج']:
                    print("👋 وداعاً!")
                    break
                
                if not query:
                    continue
                
                print("\n🤔 جاري البحث والتحليل...")
                result = self.ask(query, return_sources=True)
                
                print(f"\n💬 الإجابة:\n{result['answer']}\n")
                
                show_sources = input("📚 عرض المصادر؟ (y/n): ").strip().lower()
                if show_sources == 'y':
                    print("\n" + "="*60)
                    print("المصادر المستخدمة:")
                    print("="*60)
                    for i, doc in enumerate(result['source_documents'][:5], 1):
                        chunk_id = doc.metadata.get('chunk_id', 'context')
                        source_type = doc.metadata.get('source', 'retrieval')
                        print(f"\n[Source {i}] {chunk_id} ({source_type})")
                        print(doc.page_content[:400])
                        print("-" * 40)
                
                print("\n" + "-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 وداعاً!")
                break
            except Exception as e:
                print(f"\n❌ خطأ: {str(e)}\n")


# === Usage ===
if __name__ == "__main__":
    # Initialize with optimized parameters
    chatbot = EnhancedPharaonicRAG(
        model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        chunk_size=500,          # Medium chunks for good balance
        chunk_overlap=150,       # 30% overlap for continuity
        parent_chunk_size=2000   # Large parents for context
    )
    
    file_path = "data/pharaonic/pharaonic_info.txt"
    
    if os.path.exists(file_path):
        chatbot.build_knowledge_base(file_path)
        chatbot.save_knowledge_base()
    else:
        try:
            chatbot.load_knowledge_base()
        except:
            print(f"❌ File not found: {file_path}")
            exit(1)
    
    # Optional: Test retrieval
    # print("\n" + "="*60)
    # print("اختبار النظام - SYSTEM TEST")
    # print("="*60)
    # chatbot.test_retrieval("ما هو دور المرأة في مصر القديمة؟")
    
    # Start chat
    chatbot.chat()