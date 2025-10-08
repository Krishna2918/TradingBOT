"""
Simple HTTP server to serve AI log files
Access AI logs at: http://127.0.0.1:8052
"""

import http.server
import socketserver
import os
from pathlib import Path

class LogFileHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Handle log file requests
        if self.path.startswith('/logs/'):
            log_file = self.path[1:]  # Remove leading slash
            if Path(log_file).exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.wfile.write(content.encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Log file not found')
        else:
            # Serve the HTML file
            if self.path == '/' or self.path == '/index.html':
                self.path = '/open_ai_logs.html'
            super().do_GET()

def main():
    PORT = 8052
    
    # Change to the project directory
    os.chdir(Path(__file__).parent)
    
    with socketserver.TCPServer(("", PORT), LogFileHandler) as httpd:
        print("=" * 60)
        print("🤖 AI Trading Logs Server")
        print("=" * 60)
        print(f"🌐 AI Logs URL: http://127.0.0.1:{PORT}")
        print(f"📊 Dashboard URL: http://127.0.0.1:8051")
        print()
        print("📋 Available Log Files:")
        print("   📄 Activity Log: http://127.0.0.1:8052/logs/ai_activity.log")
        print("   📈 Trades Log: http://127.0.0.1:8052/logs/ai_trades.log")
        print("   📡 Signals Log: http://127.0.0.1:8052/logs/ai_signals.log")
        print("   🧠 Decisions Log: http://127.0.0.1:8052/logs/ai_decisions.log")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nAI Logs Server stopped.")

if __name__ == "__main__":
    main()
