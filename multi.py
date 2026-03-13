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
from typing import List, Dict
import os
from dotenv import load_dotenv
from pathlib import Path
import pickle

load_dotenv()


class MultiEraEgyptianRAG:
    
    ERAS = {
        '1': {'name': 'pharaonic', 'name_ar': 'الفرعوني', 'emoji': '🏛️'},
        '2': {'name': 'medieval', 'name_ar': 'الوسطى', 'emoji': '🕌'},
        '3': {'name': 'greek', 'name_ar': 'اليوناني', 'emoji': '🏺'},
        '4': {'name': 'roman', 'name_ar': 'الروماني', 'emoji': '🏟️'}
    }
    
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
                 chunk_size=500, chunk_overlap=150, parent_chunk_size=2000):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        
        self.era_data = {
            'pharaonic': self._init_era_structure(),
            'medieval': self._init_era_structure(),
            'greek': self._init_era_structure(),
            'roman': self._init_era_structure()
        }
        
        self.current_era = None
        self.embeddings = None
        
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2000
        )
        
    def _init_era_structure(self):
        return {
            'vectorstore': None,
            'bm25_retriever': None,
            'ensemble_retriever': None,
            'qa_chain': None,
            'child_documents': [],
            'parent_documents': [],
            'chunk_to_parent_map': {},
            'loaded': False
        }
    
    def load_text_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def build_era_knowledge_base(self, era_name, file_path):
        if era_name not in self.era_data:
            raise ValueError(f"عصر غير معروف: {era_name}")
        
        era = self.era_data[era_name]
        emoji = next(e['emoji'] for e in self.ERAS.values() if e['name'] == era_name)
        name_ar = next(e['name_ar'] for e in self.ERAS.values() if e['name'] == era_name)
        
        print(f"\n{emoji} بناء قاعدة المعرفة للعصر {name_ar}...")
        print(f"📚 تحميل النص من {file_path}...")
        
        text = self.load_text_file(file_path)
        
        print(f"✂️  تقسيم النص بنظام Parent-Child...")
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "، ", "؛ ", " "],
            keep_separator=True
        )
        
        parent_docs = parent_splitter.create_documents([text])
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "، ", "؛ ", " "],
            keep_separator=True
        )
        
        for parent_idx, parent_doc in enumerate(parent_docs):
            children = child_splitter.split_documents([parent_doc])
            
            for child_idx, child in enumerate(children):
                child.metadata['parent_idx'] = parent_idx
                child.metadata['child_idx'] = child_idx
                child.metadata['chunk_id'] = f"{era_name}_p{parent_idx}_c{child_idx}"
                child.metadata['era'] = era_name
                
                era['child_documents'].append(child)
                era['chunk_to_parent_map'][child.metadata['chunk_id']] = parent_doc
        
        era['parent_documents'] = parent_docs
        
        print(f"   ✅ {len(parent_docs)} parent chunks")
        print(f"   ✅ {len(era['child_documents'])} child chunks")
        
        print(f"🧠 إعداد نظام البحث الهجين...")
        
        era['bm25_retriever'] = BM25Retriever.from_documents(era['child_documents'])
        era['bm25_retriever'].k = 6
        
        if self.embeddings is None:
            print(f"   📊 إنشاء نموذج Embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        era['vectorstore'] = FAISS.from_documents(era['child_documents'], self.embeddings)
        faiss_retriever = era['vectorstore'].as_retriever(search_kwargs={"k": 6})
        
        era['ensemble_retriever'] = EnsembleRetriever(
            retrievers=[era['bm25_retriever'], faiss_retriever],
            weights=[0.4, 0.6]
        )
        
        self._setup_era_qa_chain(era_name)
        
        era['loaded'] = True
        print(f"✅ قاعدة معرفة العصر {name_ar} جاهزة!\n")
    
    def _expand_context(self, era_name, retrieved_docs, num_neighbors=1):
        era = self.era_data[era_name]
        expanded_docs = []
        seen_chunks = set()
        
        for doc in retrieved_docs:
            parent_idx = doc.metadata.get('parent_idx')
            child_idx = doc.metadata.get('child_idx')
            chunk_id = doc.metadata.get('chunk_id')
            
            if chunk_id in seen_chunks:
                continue
            
            neighbors = []
            for offset in range(-num_neighbors, num_neighbors + 1):
                neighbor_id = f"{era_name}_p{parent_idx}_c{child_idx + offset}"
                
                for child_doc in era['child_documents']:
                    if child_doc.metadata.get('chunk_id') == neighbor_id:
                        if neighbor_id not in seen_chunks:
                            neighbors.append(child_doc)
                            seen_chunks.add(neighbor_id)
                        break
            
            if neighbors:
                expanded_docs.extend(neighbors)
            else:
                parent_doc = era['chunk_to_parent_map'].get(chunk_id)
                if parent_doc:
                    expanded_doc = Document(
                        page_content=parent_doc.page_content,
                        metadata={'source': 'parent_context', **doc.metadata}
                    )
                    expanded_docs.append(expanded_doc)
        
        return expanded_docs[:8]
    
    def _setup_era_qa_chain(self, era_name):
        era = self.era_data[era_name]
        name_ar = next(e['name_ar'] for e in self.ERAS.values() if e['name'] == era_name)
        
        template = f"""أنت مساعد خبير في التاريخ المصري - العصر {name_ar}.

⚠️ قواعد الإجابة:
1. استخدم المعلومات من السياق الموسع أدناه
2. السياق يحتوي على أجزاء مترابطة - استخدمها معاً
3. أجب بشكل مفصل وشامل عن العصر {name_ar}
4. إذا لم تجد المعلومة في السياق، قل: "هذه المعلومة غير موجودة في قاعدة بيانات العصر {name_ar}"
5. اربط المعلومات من أجزاء مختلفة لإعطاء إجابة كاملة

السياق الموسع:
==================
{{context}}
==================

السؤال: {{question}}

الإجابة:"""

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        class ContextExpandedRetriever(BaseRetriever):
            base_retriever: object
            expand_fn: object
            
            class Config:
                arbitrary_types_allowed = True
            
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
            ) -> List[Document]:
                base_docs = self.base_retriever.get_relevant_documents(query)
                return self.expand_fn(base_docs, num_neighbors=1)
        
        expanded_retriever = ContextExpandedRetriever(
            base_retriever=era['ensemble_retriever'],
            expand_fn=lambda docs, num_neighbors: self._expand_context(era_name, docs, num_neighbors)
        )
        
        era['qa_chain'] = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=expanded_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def save_all_eras(self, base_dir="knowledge_base"):
        Path(base_dir).mkdir(exist_ok=True)
        
        for era_name, era in self.era_data.items():
            if not era['loaded']:
                continue
            
            era_dir = f"{base_dir}/{era_name}"
            Path(era_dir).mkdir(exist_ok=True)
            
            era['vectorstore'].save_local(f"{era_dir}/faiss_index")
            
            with open(f"{era_dir}/documents.pkl", 'wb') as f:
                pickle.dump({
                    'child_documents': era['child_documents'],
                    'parent_documents': era['parent_documents'],
                    'chunk_to_parent_map': era['chunk_to_parent_map']
                }, f)
        
        with open(f"{base_dir}/config.pkl", 'wb') as f:
            pickle.dump({
                'model_name': self.model_name,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'parent_chunk_size': self.parent_chunk_size
            }, f)
        
        print(f"💾 تم حفظ جميع العصور في {base_dir}/")
    
    def load_all_eras(self, base_dir="knowledge_base"):
        with open(f"{base_dir}/config.pkl", 'rb') as f:
            config = pickle.load(f)
            self.model_name = config['model_name']
            self.chunk_size = config['chunk_size']
            self.chunk_overlap = config['chunk_overlap']
            self.parent_chunk_size = config['parent_chunk_size']
        
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        for era_name in self.era_data.keys():
            era_dir = f"{base_dir}/{era_name}"
            
            if not os.path.exists(era_dir):
                continue
            
            era = self.era_data[era_name]
            
            with open(f"{era_dir}/documents.pkl", 'rb') as f:
                data = pickle.load(f)
                era['child_documents'] = data['child_documents']
                era['parent_documents'] = data['parent_documents']
                era['chunk_to_parent_map'] = data['chunk_to_parent_map']
            
            era['vectorstore'] = FAISS.load_local(
                f"{era_dir}/faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            era['bm25_retriever'] = BM25Retriever.from_documents(era['child_documents'])
            era['bm25_retriever'].k = 6
            
            faiss_retriever = era['vectorstore'].as_retriever(search_kwargs={"k": 6})
            era['ensemble_retriever'] = EnsembleRetriever(
                retrievers=[era['bm25_retriever'], faiss_retriever],
                weights=[0.4, 0.6]
            )
            
            self._setup_era_qa_chain(era_name)
            era['loaded'] = True
        
        print(f"✅ تم تحميل جميع العصور المتاحة!")
    
    def switch_era(self, era_name):
        if era_name not in self.era_data:
            return False
        
        if not self.era_data[era_name]['loaded']:
            return False
        
        self.current_era = era_name
        return True
    
    def ask(self, query, era_name=None, return_sources=False):
        if era_name:
            self.switch_era(era_name)
        
        if self.current_era is None:
            raise Exception("لم يتم اختيار عصر. استخدم switch_era() أولاً")
        
        era = self.era_data[self.current_era]
        
        if era['qa_chain'] is None:
            raise Exception(f"قاعدة معرفة العصر {self.current_era} غير جاهزة")
        
        result = era['qa_chain'].invoke({"query": query})
        
        response = {
            'answer': result['result'],
            'source_documents': result['source_documents'],
            'era': self.current_era
        }
        
        return response if return_sources else response['answer']
    
    def get_loaded_eras(self):
        return [name for name, era in self.era_data.items() if era['loaded']]
    
    def display_main_menu(self):
        print("\n" + "="*70)
        print("🏺 نظام RAG للتاريخ المصري عبر العصور")
        print("="*70)
        
        loaded_eras = self.get_loaded_eras()
        
        if not loaded_eras:
            print("❌ لم يتم تحميل أي عصر!")
            print("استخدم build_era_knowledge_base() أولاً")
            return
        
        print("\n📚 العصور المتاحة:")
        for key, info in self.ERAS.items():
            if info['name'] in loaded_eras:
                era = self.era_data[info['name']]
                status = "✅" if self.current_era == info['name'] else "  "
                print(f"{status} [{key}] {info['emoji']} {info['name_ar']}")
                print(f"      └─ {len(era['child_documents'])} chunks")
        
        print("\n🎯 الأوامر:")
        print("   [1-4] - اختر عصراً")
        print("   'قائمة' - عرض القائمة")
        print("   'خروج' - إنهاء")
        print("="*70)
    
    def chat(self):
        self.display_main_menu()
        
        while True:
            try:
                if self.current_era:
                    emoji = next(e['emoji'] for e in self.ERAS.values() if e['name'] == self.current_era)
                    name_ar = next(e['name_ar'] for e in self.ERAS.values() if e['name'] == self.current_era)
                    prompt = f"\n{emoji} [{name_ar}] أنت: "
                else:
                    prompt = "\n❓ اختر عصراً أولاً: "
                
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'خروج', 'q']:
                    print("\n👋 شكراً لاستخدامك النظام. وداعاً!")
                    break
                
                if user_input in ['قائمة', 'menu', 'm']:
                    self.display_main_menu()
                    continue
                
                if user_input in self.ERAS:
                    era_info = self.ERAS[user_input]
                    if self.switch_era(era_info['name']):
                        print(f"\n✅ تم التبديل إلى العصر {era_info['emoji']} {era_info['name_ar']}")
                    else:
                        print(f"\n❌ العصر {era_info['name_ar']} غير محمّل!")
                    continue
                
                if self.current_era is None:
                    print("⚠️  اختر عصراً أولاً! (1-4)")
                    continue
                
                print("\n🔍 جاري البحث...")
                result = self.ask(user_input, return_sources=True)
                
                print(f"\n💬 {result['answer']}\n")
                
                show = input("📚 عرض المصادر؟ (y/n): ").strip().lower()
                if show == 'y':
                    print("\n" + "="*60)
                    print("المصادر:")
                    print("="*60)
                    for i, doc in enumerate(result['source_documents'][:3], 1):
                        print(f"\n[{i}] {doc.metadata.get('chunk_id', 'N/A')}")
                        print(doc.page_content[:300] + "...")
                        print("-"*40)
                
            except KeyboardInterrupt:
                print("\n\n👋 وداعاً!")
                break
            except Exception as e:
                print(f"\n❌ خطأ: {str(e)}")


if __name__ == "__main__":
    chatbot = MultiEraEgyptianRAG(
        chunk_size=500,
        chunk_overlap=150,
        parent_chunk_size=2000
    )
    
    era_files = {
        'pharaonic': 'data/pharaonic/pharaonic_info.txt',
        'medieval': 'data/medieval/medieval_info.txt',
        'greek': 'data/greek/greek_info.txt',
        'roman': 'data/roman/roman_info.txt'
    }
    
    print("🚀 بدء بناء قواعد المعرفة...\n")
    
    for era_name, file_path in era_files.items():
        if os.path.exists(file_path):
            chatbot.build_era_knowledge_base(era_name, file_path)
        else:
            print(f"⚠️  ملف {era_name} غير موجود: {file_path}")
    
    chatbot.save_all_eras()
    
    if chatbot.get_loaded_eras():
        chatbot.switch_era(chatbot.get_loaded_eras()[0])
    
    chatbot.chat()