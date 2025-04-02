import os
from flask import Flask, render_template, request, jsonify, session
from chat_chain import ChatRAGChain
from embedding import DocumentProcessor
from llm_model import llm_deepseek
from rag_model import MyEmbeddings
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 用于session加密
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class Frontend:
    def __init__(self):
        self.chat_chain = None
        self.embedding = None
        self.llm_model = None
        self.rag_model = None
        self.chat_history = []  # 添加聊天历史记录列表
        
    def setup_components(self, 
                        pdf_path: str,
                        temperature: float = 0.7,
                        max_tokens: int = 2000,
                        chunk_size: int = 500,
                        chunk_overlap: int = 50):
        """Initialize all components with user-specified parameters"""
        try:
            # Initialize embedding
            self.embedding = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Process the PDF document
            self.embedding.process_document(pdf_path)
            
            # Initialize LLM model
            self.llm_model = llm_deepseek()
            
            # Initialize RAG model
            self.rag_model = MyEmbeddings()
            
            # Initialize chat chain
            self.chat_chain = ChatRAGChain(
                llm_model=self.llm_model,
                embedding_model=self.rag_model
            )
            
            # 重置聊天历史记录
            self.chat_history = []
            
            return True, "Components initialized successfully!"
            
        except Exception as e:
            return False, f"Error initializing components: {str(e)}"
    
    def get_response(self, user_input: str):
        """Get response from the chat chain"""
        try:
            print(f"Getting response for input: {user_input}")  # Debug log
            
            # 检查chat_chain是否已初始化
            if self.chat_chain is None:
                print("Error: chat_chain is not initialized")  # Debug log
                return False, "Chat system is not initialized. Please upload a PDF file first."
            
            # 使用流式输出
            print("Answer: ", end="", flush=True)
            response_chunks = []
            for chunk in self.chat_chain.chat(user_input, chat_history=self.chat_history, stream=True):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    response_chunks.append(chunk)
            print("\n")  # 换行
            
            # 将所有chunks组合成完整响应
            final_response = ''.join(response_chunks)
            print(f"Final response: {final_response}")  # Debug log
            
            # 更新聊天历史记录
            self.chat_history.append((user_input, final_response))
            # 保持最多10条历史记录
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            return True, final_response
            
        except Exception as e:
            print(f"Error in get_response: {str(e)}")  # Debug log
            import traceback
            print(f"Traceback: {traceback.format_exc()}")  # Debug log
            return False, f"Error getting response: {str(e)}"

# 创建全局Frontend实例
frontend = Frontend()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取其他参数
        temperature = float(request.form.get('temperature', 0.7))
        max_tokens = int(request.form.get('max_tokens', 2000))
        chunk_size = int(request.form.get('chunk_size', 500))
        chunk_overlap = int(request.form.get('chunk_overlap', 50))
        
        # 初始化组件
        success, message = frontend.setup_components(
            pdf_path=filepath,
            temperature=temperature,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return jsonify({
            'success': success,
            'message': message
        })
    
    return jsonify({'success': False, 'message': 'Invalid file type'})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({'success': False, 'message': 'No message provided'})
    
    success, response = frontend.get_response(user_input)
    return jsonify({
        'success': success,
        'message': response
    })

if __name__ == '__main__':
    app.run(debug=True)
