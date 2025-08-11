Autonomous AI Agents System

A sophisticated multi-agent AI system with Retrieval-Augmented Generation (RAG) capabilities and a web interface, featuring complete event loop conflict resolution.

Key Features

- Event Loop Fixed Architecture
  - No more "Cannot run the event loop while another loop is running" errors
  - Thread pool execution for isolated request handling
  - Thread-safe operations implemented

- Advanced RAG Integration
  - Vector database with semantic search
  - Context-aware response generation
  - Dynamic knowledge retrieval

- Specialized Autonomous Agents
  - Researcher Agent: Information gathering and analysis
  - Analyst Agent: Data processing and performance metrics
  - Coordinator Agent: Workflow orchestration

- Modern Web Interface
  - Real-time chat interface
  - System monitoring dashboard
  - Agent selection controls

System Architecture

The system consists of:
- Web Interface
- MultiAgentOrchestrator
  - Researcher Agent
  - Analyst Agent
  - Coordinator Agent
    - AdvancedRAGEngine
      - CloudVectorStore
      - Cloud LLMService
<img width="1710" height="935" alt="Screenshot 2025-08-11 at 7 08 59 PM" src="https://github.com/user-attachments/assets/a4521019-3fc2-4a89-8400-9fc52f477003" />
<img width="1710" height="944" alt="Screenshot 2025-08-11 at 7 13 15 PM" src="https://github.com/user-attachments/assets/e7a19fd3-3a56-40d4-aab9-5465ead76895" />

     
    

Installation

1. Clone the repository:
   git clone https://github.com/prathampreet2002/autonomous-agents-using-cloud.git
   cd autonomous-agents

2. Install dependencies:
   pip install -r requirements.txt

Usage

Running the Web Interface (Recommended):
python autonomous_agents.py
Select option 1 when prompted to launch the full web interface.

Running Backend Demo:
python autonomous_agents.py
Select option 2 when prompted for backend-only demo.

Configuration

The system is pre-configured with:
- Default port: 8080 (change in WebInterface class)
- Pre-loaded knowledge base with 5 technical documents
- Three specialized agents with distinct capabilities

Technical Highlights

- Fixed Event Loop Implementation
  - Thread pool execution prevents event loop conflicts
  - Each request gets isolated event loop
  - Proper async/sync separation

- Enhanced RAG Engine
  - Dynamic context retrieval
  - Semantic ranking with metadata filtering
  - Embedding caching for performance

- Comprehensive Monitoring
  - Real-time system metrics
  - Agent performance tracking
  - LLM usage statistics

Example Queries

Try these in the web interface:
- "Explain how RAG improves LLM accuracy"
- "Analyze multi-agent system performance"
- "Design a workflow for document processing"
- "Compare vector database solutions"

Troubleshooting

If you encounter port conflicts:
Change the port number in WebInterface initialization
web_interface = WebInterface(orchestrator, port=8081)

For other issues:
- Check Python version (requires 3.8+)
- Ensure no other process is using the port
- Verify all dependencies are installed

License

MIT License - Free for academic and commercial use

Note: The system automatically opens your web browser to http://localhost:8080 when started. If this doesn't work, please navigate manually.
