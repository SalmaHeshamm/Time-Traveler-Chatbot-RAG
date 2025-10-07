import os, random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

class RAGEngine:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.vstores = {}
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    def _read_file(self, path):
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def build_vectorstore(self, era):
        info_path = os.path.join(self.data_dir, era, f"{era}_info.txt")
        stories_path = os.path.join(self.data_dir, era, f"{era}_stories.txt")
        texts = []
        if os.path.exists(info_path):
            texts.append(self._read_file(info_path))
        if os.path.exists(stories_path):
            texts.append(self._read_file(stories_path))
        raw = "\n\n".join(texts)
        if not raw.strip():
            return None
        chunks = self.splitter.split_text(raw)
        # create FAISS vectorstore using OpenAI embeddings (requires OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings()
        vstore = FAISS.from_texts(chunks, embeddings)
        self.vstores[era] = {"vstore": vstore, "chunks": chunks}
        return self.vstores[era]

    def get_answer(self, era, query, k=3):
        if era not in self.vstores:
            self.build_vectorstore(era)
        if era not in self.vstores or self.vstores[era] is None:
            # fallback: substring search in info file
            info_path = os.path.join(self.data_dir, era, f"{era}_info.txt")
            data = self._read_file(info_path)
            if query in data:
                start = data.find(query)
                return data[start:start+600]
            return None
        vstore = self.vstores[era]["vstore"]
        retriever = vstore.as_retriever(search_kwargs={"k": k})
        try:
            llm = ChatOpenAI(temperature=0.3)
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
            res = qa.run(query)
            return res
        except Exception as e:
            docs = retriever.get_relevant_documents(query)
            return "\n\n".join([d.page_content for d in docs])

    def get_story(self, era, persona):
        path = os.path.join(self.data_dir, era, f"{era}_stories.txt")
        if not os.path.exists(path):
            return "لا توجد حكايات مضافة لهذا العصر."
        data = self._read_file(path)
        parts = [p.strip() for p in data.split("\n\n") if p.strip()]
        persona = persona.lower()
        for p in parts:
            if persona in p.lower():
                return p
        return random.choice(parts) if parts else "لا توجد حكايات."
