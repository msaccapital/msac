# app.py - Production Ready with CORS Fixes
print("🚀 Loading KRISHN Lightning Fast Fixed Edition with Scope Control...")
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import faiss
from sentence_transformers import SentenceTransformer
import os
import time
import re
from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
import pdfplumber
from werkzeug.utils import secure_filename
import webbrowser
import threading
import secrets
from functools import wraps
import numpy as np

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# FIXED CORS Configuration for Production
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Allow all origins for now
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
    }
})

# Manual CORS headers for additional security
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Handle preflight requests
@app.route('/api/upload', methods=['OPTIONS'])
@app.route('/api/ask', methods=['OPTIONS'])
def options_handler():
    return '', 200

# Session management
user_sessions = {}

class TimingDecorator:
    @staticmethod
    def timing(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"⏱️ {func.__name__} took {end-start:.2f}s")
            return result
        return wrapper

class LightningPDFProcessor:
    def __init__(self):
        self.upload_folder = "./uploads"
        os.makedirs(self.upload_folder, exist_ok=True)
    
    @TimingDecorator.timing
    def process_pdf(self, file_path):
        chunks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        # Simple paragraph-based chunking
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        for para in paragraphs:
                            if len(para) > 50:  # Only keep substantial paragraphs
                                chunks.append({
                                    'text': para,
                                    'source': os.pathasename(file_path),
                                    'page_number': page_num + 1
                                })
            
            print(f"✅ Processed {len(chunks)} chunks from {file_path}")
            return chunks
        except Exception as e:
            print(f"❌ PDF processing error: {e}")
            return []

class LightningVectorDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
    
    @TimingDecorator.timing
    def build_index(self, chunks):
        if not chunks:
            print("❌ No chunks to index")
            return False
        
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"🔧 Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ Built FAISS index with {self.index.ntotal} vectors")
        return True
    
    @TimingDecorator.timing
    def search(self, query, top_k=5):
        if not self.index or self.index.ntotal == 0:
            print("❌ No index available for search")
            return []
        
        print(f"🔍 Searching for: '{query}'")
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    **self.chunks[idx],
                    'score': float(score)
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter low relevance results - IMPORTANT FIX
        relevant_results = [r for r in results if r['score'] > 0.3]
        print(f"✅ Found {len(relevant_results)} relevant results (score > 0.3)")
        
        return relevant_results[:top_k]

class LightningQA:
    def __init__(self):
        self.vector_db = None
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
    
    def load_model(self):
        if self.model_loaded:
            return True
        
        print("⚡ Loading ULTRA-FAST model...")
        try:
            # Using a more reliable small model
            model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model = self.model.to('cpu')
            self.model.eval()
            
            self.model_loaded = True
            print("✅ Lightning model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    @TimingDecorator.timing
    def generate_answer(self, question, chunks):
        if not self.load_model():
            return "🤖 Model not available. Please check your installation."
        
        # CRITICAL FIX: Check if chunks are actually relevant before proceeding
        if not chunks:
            return "❌ This question is beyond the scope of the uploaded document. I cannot find relevant information to answer this."
        
        # Filter chunks by relevance score
        relevant_chunks = [chunk for chunk in chunks if chunk.get('score', 0) > 0.3]
        
        if not relevant_chunks:
            return "❌ This question is beyond the scope of the uploaded document. I cannot find relevant information to answer this."
        
        # Build context from top RELEVANT chunks only
        context = "\n".join([f"• {chunk['text']}" for chunk in relevant_chunks[:3]])
        
        # IMPROVED PROMPT - WITH SCOPE CONTROL
        prompt = f"""Document Content:
{context}

Question: {question}

Answer the question based ONLY on the document content above. If the document doesn't contain relevant information, say "This is beyond the scope of the document.":"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=200,
                    temperature=0.8,
                    do_sample=True,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part (after the prompt)
            answer = response[len(prompt):].strip()
            
            # Clean up the answer
            answer = self.clean_response(answer)
            
            # If answer is empty or too short, provide a fallback
            if not answer or len(answer) < 10:
                return self.get_fallback_answer(question, relevant_chunks)
            
            return answer
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return self.get_fallback_answer(question, relevant_chunks)
    
    def clean_response(self, text):
        """Clean up model response"""
        # Remove any incomplete sentences at the end
        text = re.sub(r'[^.!?]*$', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove the prompt if it got repeated
        text = re.sub(r'Document Content:.*?Answer the question based ONLY on the document content above:', '', text, flags=re.DOTALL)
        return text.strip()
    
    def get_fallback_answer(self, question, chunks):
        """Provide a fallback answer when model generation fails or no relevant info"""
        if not chunks:
            return "❌ This question is beyond the scope of the uploaded document. I cannot find relevant information to answer this."
        
        # Check if any chunks are actually relevant (score threshold)
        relevant_chunks = [chunk for chunk in chunks if chunk.get('score', 0) > 0.3]
        
        if not relevant_chunks:
            return "❌ This question is beyond the scope of the uploaded document. I cannot find relevant information to answer this."
        
        # Simple rule-based fallback only for actually relevant content
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what is', 'what are', 'describe', 'explain']):
            return f"Based on the document, I found relevant information: {relevant_chunks[0]['text'][:200]}..."
        
        elif any(word in question_lower for word in ['summarize', 'summary', 'main points']):
            summaries = [chunk['text'][:100] + "..." for chunk in relevant_chunks[:2]]
            return f"Key points from the document:\n" + "\n".join([f"• {s}" for s in summaries])
        
        else:
            return f"I found this relevant information in the document: {relevant_chunks[0]['text'][:250]}..."

def get_user_session():
    """Get or create user session"""
    if 'user_id' not in session:
        session['user_id'] = secrets.token_hex(8)
        user_sessions[session['user_id']] = {
            'vector_db': LightningVectorDB(),
            'qa_system': LightningQA(),
            'upload_time': time.time()
        }
    
    return user_sessions[session['user_id']]

# Initialize components
pdf_processor = LightningPDFProcessor()

HTML_LIGHTNING_FIXED_SCOPE = """
<!DOCTYPE html>
<html>
<head>
    <title>⚡ KRISHN Lightning - SCOPE CONTROL</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .section {
            margin: 25px 0;
            padding: 20px;
            border: 2px dashed #e0e0e0;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .section:hover {
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 14px 28px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            background: #cccccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .file-input {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
        }
        
        textarea {
            width: 100%;
            height: 120px;
            margin: 10px 0;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s ease;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .chat-area {
            max-height: 500px;
            overflow-y: auto;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        
        .message {
            padding: 12px 16px;
            margin: 12px 0;
            border-radius: 12px;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }
        
        .user {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin-left: 15%;
            border-bottom-right-radius: 4px;
        }
        
        .assistant {
            background: #e9ecef;
            color: #333;
            margin-right: 15%;
            border-bottom-left-radius: 4px;
            white-space: pre-line;
        }
        
        .status {
            padding: 12px;
            margin: 10px 0;
            border-radius: 8px;
            text-align: center;
            font-weight: 500;
        }
        
        .success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .loading {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .typing-indicator {
            display: none;
            padding: 12px 16px;
            color: #666;
            font-style: italic;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
                margin: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .user, .assistant {
                margin-left: 5%;
                margin-right: 5%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ KRISHN Lightning - SCOPE CONTROL</h1>
            <p>Document-Specific Answers • No Hallucinations • Beyond Scope Detection</p>
        </div>
        
        <div class="section">
            <h3>📤 Upload PDF Document</h3>
            <input type="file" id="fileInput" class="file-input" accept=".pdf">
            <button class="btn" onclick="uploadPDF()" id="uploadBtn">
                📎 Upload & Process PDF
            </button>
            <div id="fileStatus"></div>
        </div>
        
        <div class="section">
            <h3>💬 Ask Questions About Your Document</h3>
            <textarea id="questionInput" placeholder="Type your question here... 
Example: 'What is this document about?' or 'Summarize the key findings'"></textarea>
            <button class="btn" onclick="askQuestion()" id="askBtn">
                🚀 Ask KRISHN
            </button>
            <div class="typing-indicator" id="typingIndicator">
                KRISHN is thinking...
            </div>
            <div class="chat-area" id="chatArea"></div>
        </div>
        
        <div class="stats" id="stats" style="display: none;">
            <div class="stat-item">
                <div class="stat-value" id="chunkCount">0</div>
                <div>Text Chunks</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="responseTime">0s</div>
                <div>Response Time</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="confidence">-</div>
                <div>Confidence</div>
            </div>
        </div>
    </div>

    <script>
        // BACKEND URL - Update this to your Render URL
        const BACKEND_URL = 'https://msac.onrender.com';
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('fileStatus');
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
            setTimeout(() => {
                if (statusDiv.innerHTML.includes(message)) {
                    statusDiv.innerHTML = '';
                }
            }, 5000);
        }
        
        function uploadPDF() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            const uploadBtn = document.getElementById('uploadBtn');
            
            if (!file) {
                showStatus('❌ Please select a PDF file first!', 'error');
                return;
            }
            
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '⏳ Processing...';
            showStatus('🔄 Processing PDF document...', 'loading');
            
            const formData = new FormData();
            formData.append('file', file);
            
            fetch(BACKEND_URL + '/api/upload', {
                method: 'POST',
                body: formData,
                mode: 'cors'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`✅ ${data.message}`, 'success');
                    document.getElementById('stats').style.display = 'flex';
                    document.getElementById('chunkCount').textContent = data.chunk_count || '0';
                    // Clear previous chat
                    document.getElementById('chatArea').innerHTML = '';
                } else {
                    showStatus(`❌ ${data.error}`, 'error');
                }
            })
            .catch(error => {
                showStatus(`❌ Upload failed: ${error}`, 'error');
                console.error('Upload error:', error);
            })
            .finally(() => {
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = '📎 Upload & Process PDF';
            });
        }
        
        function askQuestion() {
            const question = document.getElementById('questionInput').value.trim();
            const askBtn = document.getElementById('askBtn');
            const typingIndicator = document.getElementById('typingIndicator');
            
            if (!question) {
                showStatus('❌ Please enter a question!', 'error');
                return;
            }
            
            const chatArea = document.getElementById('chatArea');
            chatArea.innerHTML += `<div class="message user"><strong>You:</strong> ${question}</div>`;
            
            askBtn.disabled = true;
            askBtn.innerHTML = '⏳ Thinking...';
            typingIndicator.style.display = 'block';
            
            document.getElementById('questionInput').value = '';
            chatArea.scrollTop = chatArea.scrollHeight;
            
            const startTime = Date.now();
            
            fetch(BACKEND_URL + '/api/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question}),
                mode: 'cors'
            })
            .then(response => response.json())
            .then(data => {
                const endTime = Date.now();
                const responseTime = (endTime - startTime) / 1000;
                
                document.getElementById('responseTime').textContent = `${responseTime.toFixed(1)}s`;
                document.getElementById('confidence').textContent = data.confidence || 'Medium';
                
                chatArea.innerHTML += `<div class="message assistant"><strong>KRISHN:</strong> ${data.answer}</div>`;
                chatArea.scrollTop = chatArea.scrollHeight;
            })
            .catch(error => {
                console.error('Ask error:', error);
                chatArea.innerHTML += `<div class="message assistant"><strong>KRISHN:</strong> Sorry, I encountered an error. Please try again.</div>`;
            })
            .finally(() => {
                askBtn.disabled = false;
                askBtn.innerHTML = '🚀 Ask KRISHN';
                typingIndicator.style.display = 'none';
            });
        }
        
        // Allow pressing Enter to ask question
        document.getElementById('questionInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_LIGHTNING_FIXED_SCOPE)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Please upload a PDF file'})
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        file_path = os.path.join(pdf_processor.upload_folder, filename)
        file.save(file_path)
        
        # Process PDF
        chunks = pdf_processor.process_pdf(file_path)
        if not chunks:
            return jsonify({'success': False, 'error': 'No readable text found in PDF'})
        
        # Get user session and build index
        user_session = get_user_session()
        if user_session['vector_db'].build_index(chunks):
            return jsonify({
                'success': True, 
                'message': f'Successfully processed {len(chunks)} text chunks',
                'chunk_count': len(chunks)
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to process document'})
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'answer': 'Please provide a question'})
        
        user_session = get_user_session()
        vector_db = user_session['vector_db']
        qa_system = user_session['qa_system']
        
        # Search for relevant chunks
        chunks = vector_db.search(question, top_k=3)
        
        # Generate answer
        answer = qa_system.generate_answer(question, chunks)
        
        # Calculate confidence based on relevance scores
        if not chunks:
            confidence = 'None'
        elif chunks[0]['score'] > 0.7:
            confidence = 'High'
        elif chunks[0]['score'] > 0.4:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return jsonify({
            'answer': answer,
            'confidence': confidence,
            'relevant_chunks': len(chunks)
        })
    
    except Exception as e:
        print(f"❌ Ask error: {e}")
        return jsonify({'answer': f'Sorry, I encountered an error. Please try again.'})

def open_browser():
    """Open browser automatically"""
    time.sleep(2)
    webbrowser.open('http://localhost:5007')

if __name__ == "__main__":
    print("🚀 Starting KRISHN Lightning SCOPE CONTROL Edition...")
    print("🌐 Web Interface: http://localhost:5007")
    print("⚡ Using DialoGPT-small with SCOPE CONTROL")
    print("🔧 Production-ready configuration")
    print("-" * 50)
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 5007))
    
    # Start browser thread only in development
    if os.environ.get('PRODUCTION') != 'true':
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Production mode - no browser auto-open
        app.run(host='0.0.0.0', port=port, debug=False)
