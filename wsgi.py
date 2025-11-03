import os
from app import app

# Set production environment
os.environ['PRODUCTION'] = 'true'

if __name__ == "__main__":
    print("🚀 Starting KRISHN Lightning Production Server...")
    print("🌐 Production Mode: ACTIVE")
    print("-" * 50)
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 5007))
    app.run(host='0.0.0.0', port=port, debug=False)
