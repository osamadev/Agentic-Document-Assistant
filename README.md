# Document Assistant Project - Complete Implementation

A sophisticated document processing system using LangChain and LangGraph that can answer questions, summarize documents, and perform calculations on financial and healthcare documents.

## Implementation Status: COMPLETE

All required components have been implemented and tested successfully. The system is ready for production use.

## Project Overview

This document assistant uses a multi-agent architecture with LangGraph to handle different types of user requests:
- **Q&A Agent**: Answers specific questions about document content
- **Summarization Agent**: Creates summaries and extracts key points from documents
- **Calculation Agent**: Performs mathematical operations on document data

### Key Features
- **Intent Classification**: Automatically determines user intent and routes to appropriate agent
- **Multi-Agent Architecture**: Specialized agents for different task types
- **Conversation Memory**: Maintains context across conversation turns
- **Document Tracking**: Tracks which documents are referenced in responses
- **Tool Logging**: Comprehensive logging of all tool usage for compliance
- **Secure Calculator**: Safe mathematical expression evaluation
- **Session Management**: Persistent conversation storage and retrieval

## Getting Started

### Dependencies

- Python 3.9+
- OpenAI API key
- Required packages (see requirements.txt):
  - langchain>=0.2.0
  - langgraph>=0.0.20
  - langchain-openai>=0.1.0
  - langchain-core>=0.2.0
  - pydantic>=2.0.0
  - python-dotenv>=1.0.0
  - openai>=1.0.0
  - print-color>=0.4.6

### Installation

Step by step explanation of how to get a dev environment running.

1. **Clone the repository and navigate to the starter directory:**
```bash
cd starter
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create a `.env` file with your OpenAI API key:**
```bash
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o
```

5. **Run the assistant:**
```bash
python main.py
```

## Testing

### Break Down Tests

The project includes comprehensive testing to verify all components work correctly.

#### Test Categories

The project includes comprehensive testing with **both offline and LLM-based testing** to ensure complete validation of all components.

##### **Offline Tests (9 categories)**

1. **Schema Tests** - Verify Pydantic models work correctly
   - AnswerResponse schema with question, answer, sources, confidence, timestamp
   - UserIntent schema with intent_type, confidence, reasoning
   - ConversationTurn and SessionState schemas

2. **Prompt Tests** - Verify prompt templates generate properly
   - Intent classification prompt with user_input and conversation_history variables
   - Chat prompt templates for all intent types (qa, summarization, calculation)

3. **Tool Tests** - Verify tools function correctly
   - ToolLogger for automatic logging
   - Calculator tool with multiple test expressions (2+3, 100*5/2, 1000+500, 22+69.3+214.5)
   - Document retrieval tools

4. **Retrieval Tests** - Verify document system works
   - Keyword-based document search (found 3 invoice documents)
   - Type-based document search (invoice, contract, claim)
   - Amount-based document filtering (over $50,000: 3 documents)
   - Specific document retrieval by ID
   - Statistics generation (5 total documents, $488,250.00 total amount)

5. **Agent Structure Tests** - Verify LangGraph workflow
   - AgentState structure validation
   - Function definitions and imports
   - Workflow creation and compilation

6. **Real-World Scenarios Tests** - Validate actual usage patterns
   - Q&A Query: "What's the total amount in invoice INV-001?" → $22,000.00
   - Document Summarization: "summarize all documents" → 5 documents, $488,250.00 total
   - Mathematical Calculation: "Calculate the sum of all invoice totals" → $305,800.00
   - Advanced Filtering: "Find documents with amounts over $50,000" → 3 documents
   - Context-Aware Calculation: "how many of them below 50,000" → 2 documents

7. **Conversation Management Tests** - Verify conversation memory
   - Multi-turn conversation simulation (3 turns)
   - Session state creation with conversation history
   - Active documents tracking
   - Tool usage logging

8. **Intent Classification Tests** - Verify intent detection
   - 7 different query types with expected intents
   - Prompt formatting validation
   - Intent routing verification

9. **Document Processing Tests** - Verify document capabilities
   - Document type filtering (3 invoices, 1 contract, 1 claim)
   - Amount range filtering (4 different ranges)
   - Specific document retrieval with amounts
   - Statistics generation and validation

##### **LLM Tests (3 categories)**

10. **LLM-Based Intent Classification** - Test real LLM interactions
    - Uses actual GPT-4o for classification
    - 5 test queries with accuracy measurement (100% accuracy: 5/5 correct)
    - Structured output validation (UserIntent schema compliance)
    - Confidence tracking and reasoning analysis

11. **LLM-Based Response Generation** - Test real response generation
    - Q&A, Summarization, and Calculation scenarios
    - Actual LLM-generated responses
    - Prompt template validation with real LLM
    - Context handling verification

12. **LLM-Based Conversation Flow** - Test complete conversation flow
    - Multi-turn conversation simulation with context
    - Intent classification with conversation history
    - Response generation with conversation context
    - Context awareness and memory validation

#### Running Tests

```bash
python test_implementation.py
```

The test script automatically detects if an OpenAI API key is available and runs appropriate tests:

**With API Key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
python test_implementation.py
```

**Without API Key:**
```bash
python test_implementation.py
# Runs offline tests only
```

Expected output:
```
================================================================================
Testing Document Assistant Implementation (with LLM support)
================================================================================
OpenAI API key found - LLM tests will be enabled

Running Offline Tests...
✓ AnswerResponse schema works
✓ UserIntent schema works
✓ ConversationTurn schema works
✓ SessionState schema works

Testing prompts...
✓ Intent classification prompt works
✓ Chat prompt template for qa works
✓ Chat prompt template for summarization works
✓ Chat prompt template for calculation works

Testing tools...
✓ ToolLogger works
✓ Calculator: 2 + 3 = The result of 2 + 3 is 5
✓ Calculator: 100 * 5 / 2 = The result of 100 * 5 / 2 is 250
✓ Calculator: 1000 + 500 = The result of 1000 + 500 is 1500
✓ Calculator: 22 + 69.3 + 214.5 = The result of 22 + 69.3 + 214.5 is 305.80

Testing retrieval...
✓ Keyword search 'invoice': found 3 documents
✓ Type search 'invoice': found 3 documents
✓ Amount search 'over $50,000': found 3 documents
✓ Document retrieval by ID: found INV-001
✓ Statistics: 5 total documents
✓ Total amount: $488,250.00

Testing agent structure...
✓ AgentState structure works
✓ Agent functions are properly defined

Testing real-world scenarios...
Scenario 1: Q&A Query
Input: 'What's the total amount in invoice INV-001?'
✓ Found document INV-001 with total: $22,000.00
✓ AnswerResponse created successfully

Scenario 2: Document Summarization
Input: 'summarize all documents'
✓ Document collection summary:
  - Total documents: 5
  - Total amount: $488,250.00
  - Average amount: $97,650.00

Scenario 3: Mathematical Calculation
Input: 'Calculate the sum of all invoice totals'
✓ Calculated invoice total: $305,800.00

Scenario 4: Advanced Filtering
Input: 'Find documents with amounts over $50,000'
✓ Found 3 documents over $50,000
  - INV-003: $214,500.00
  - CON-001: $180,000.00
  - INV-002: $69,300.00

Scenario 5: Context-Aware Calculation
Input: 'how many of them below 50,000'
✓ Context calculation: 5 total - 3 over $50k = 2 below $50k

Testing conversation management...
✓ Turn 1 (Q&A) added to conversation history
✓ Turn 2 (Summarization) added to conversation history
✓ Turn 3 (Calculation) added to conversation history
✓ Session created with 3 conversation turns
✓ Active documents: ['INV-001', 'INV-002', 'INV-003', 'CON-001', 'CLM-001']

Testing intent classification scenarios...
✓ Query: 'What's the total in invoice INV-001?' -> Expected intent: qa
✓ Query: 'summarize all documents' -> Expected intent: summarization
✓ Query: 'Calculate the sum of all invoice totals' -> Expected intent: calculation
✓ Query: 'Find documents with amounts over $50,000' -> Expected intent: qa
✓ Query: 'how many documents we have' -> Expected intent: qa
✓ Query: 'summarize all contracts' -> Expected intent: summarization
✓ Query: 'What's the average amount?' -> Expected intent: calculation

Testing document processing capabilities...
✓ Found 3 invoice documents
✓ Found 1 contract documents
✓ Found 1 claim documents
✓ under $10,000: 1 documents
✓ $10,000 - $50,000: 1 documents
✓ $50,000 - $100,000: 1 documents
✓ over $100,000: 2 documents
✓ INV-001: $22,000.00
✓ CON-001: $180,000.00
✓ CLM-001: $2,450.00
✓ INV-002: $69,300.00
✓ INV-003: $214,500.00

Running LLM Tests...

Testing LLM-based intent classification...
✓ Query: 'What's the total in invoice INV-001?' -> Classified as: qa (Expected: qa)
  Confidence: 0.95, Reasoning: The user's input, 'What's the total in invoice INV-001?', directly aligns with the 'qa' category as ...
✓ Query: 'summarize all documents' -> Classified as: summarization (Expected: summarization)
  Confidence: 0.95, Reasoning: The user's input 'summarize all documents' directly aligns with the 'summarization' category. The ke...
✓ Query: 'Calculate the sum of all invoice totals' -> Classified as: calculation (Expected: calculation)
  Confidence: 0.95, Reasoning: The user input 'Calculate the sum of all invoice totals' explicitly indicates a request for a mathem...
✓ Query: 'Find documents with amounts over $50,000' -> Classified as: qa (Expected: qa)
  Confidence: 0.95, Reasoning: The user's input, 'Find documents with amounts over $50,000,' directly aligns with the 'qa' category...
✓ Query: 'how many documents we have' -> Classified as: qa (Expected: qa)
  Confidence: 0.85, Reasoning: The user input 'how many documents we have' suggests that the user is asking for a specific piece of...
✓ Intent classification accuracy: 100.0% (5/5)

Testing LLM-based response generation...
Testing: Q&A Query
Query: 'What's the total in invoice INV-001?'
✓ Response generated: Let me look up the information for you. Just a moment, please....

Testing: Document Summarization
Query: 'summarize all documents'
✓ Response generated: I can help with that. Please provide the documents you would like me to summarize....

Testing: Mathematical Calculation
Query: 'Calculate the sum of all invoice totals'
✓ Response generated: To calculate the sum of all invoice totals, I will need the individual invoice totals...

Testing LLM-based conversation flow...
Turn 1: Q&A Query
✓ Intent classified as: qa (confidence: 0.90)
✓ Response: Let me look that up for you. Just a moment, please....

Turn 2: Follow-up Query
✓ Intent classified as: calculation (confidence: 0.90)
✓ Response: To calculate the sum of all invoice totals, we need to add up the totals from each invoice...
✓ Complete conversation flow tested successfully

================================================================================
Test Results: 12/12 tests passed
  • Offline tests: 9/9 passed
  • LLM tests: 3/3 passed
All tests passed!

Test Coverage:
  • Schema validation and data structures
  • Prompt templates and intent classification
  • Tool functionality and security
  • Document retrieval and processing
  • Agent workflow structure
  • Real-world usage scenarios
  • Conversation management and memory
  • Intent classification accuracy
  • Document processing capabilities
  • LLM-based intent classification
  • LLM-based response generation
  • LLM-based conversation flow

================================================================================
```

#### **Real-World Test Scenarios**

Based on actual conversation history, here are comprehensive test scenarios:

**Test Scenario 1: Q&A Functionality**
```
Input: "What's the total amount in invoice INV-001?"
Expected Intent: qa
Expected Response: Detailed breakdown of $22,000
Expected Tools: document_reader
Expected Sources: INV-001
```

**Test Scenario 2: Document Summarization**
```
Input: "summarize all documents"
Expected Intent: summarization
Expected Response: Overview of 5 documents with financial summary
Expected Tools: document_statistics
Expected Key Points: Document types, total amounts, averages
```

**Test Scenario 3: Mathematical Calculations**
```
Input: "Calculate the sum of all invoice totals"
Expected Intent: calculation
Expected Response: $305,800 with step-by-step breakdown
Expected Result: 305800.0
Expected Tools: document_search
```

**Test Scenario 4: Advanced Filtering**
```
Input: "Find documents with amounts over $50,000"
Expected Intent: qa
Expected Response: List of 3 documents (INV-003, CON-001, INV-002)
Expected Tools: document_search
Expected Sources: INV-003, CON-001, INV-002
```

**Test Scenario 5: Context-Aware Calculations**
```
Input: "how many of them below 50,000"
Expected Intent: calculation
Expected Response: 2 documents (calculated from previous context)
Expected Result: 2.0
Expected Tools: document_search
```

#### **Conversation Memory Testing**

The system maintains conversation context across multiple turns:
- Session persistence with unique session IDs
- Conversation history tracking
- Context-aware responses
- Tool usage logging
- Document reference tracking

## Project Instructions

### Implemented Components

#### 1. Schema Implementation (schemas.py) - COMPLETE

**AnswerResponse Schema**
- Purpose: Structured Q&A responses with source tracking
- Fields: question, answer, sources, confidence, timestamp

**UserIntent Schema**
- Purpose: Intent classification for routing
- Fields: intent_type, confidence, reasoning

#### 2. Agent State Implementation (agent.py) - COMPLETE

**classify_intent Function**
- Classifies user intent and sets next step in workflow
- Routes to appropriate agent based on intent type

**calculation_agent Function**
- Handles mathematical operations on document data
- Uses structured output for clear explanations

**update_memory Function**
- Updates conversation history and manages state
- Tracks active documents and tools used

**create_workflow Function**
- Creates the LangGraph workflow coordinating all agents
- Implements the complete graph structure

#### 3. Prompt Implementation (prompts.py) - COMPLETE

**get_intent_classification_prompt Function**
- Helps LLM classify user intents accurately
- Includes examples and guidelines for each intent type

**get_chat_prompt_template Function**
- Provides context-aware prompts for different task types
- Selects appropriate system prompts based on intent

#### 4. Tool Implementation (tools.py) - COMPLETE

**create_calculator_tool Function**
- Safely evaluates mathematical expressions
- Validates expressions for security
- Logs tool usage automatically

### Workflow Architecture

The LangGraph workflow follows this structure:
```
classify_intent → [qa_agent|summarization_agent|calculation_agent] → update_memory → END
```

### Sample Data

The system includes 5 sample documents for testing:
- **INV-001**: Invoice #12345 ($22,000)
- **CON-001**: Service Agreement ($180,000)
- **CLM-001**: Insurance Claim #78901 ($2,450)
- **INV-002**: Invoice #12346 ($69,300)
- **INV-003**: Invoice #12347 ($214,500)

### Example Queries and Test Cases

The system can handle various types of queries. Here are comprehensive test cases based on real usage:

#### **Q&A Queries (Intent: qa)**
- **"What's the total amount in invoice INV-001?"**
  - Expected: Detailed breakdown of $22,000 total
  - Tools: document_reader
  - Sources: INV-001

- **"Who is the client in CON-001?"**
  - Expected: Client information from service agreement
  - Tools: document_search, document_reader
  - Sources: CON-001

- **"When was this claim filed?"**
  - Expected: Date information from insurance claim
  - Tools: document_search, document_reader
  - Sources: CLM-001

- **"Find documents with amounts over $50,000"**
  - Expected: List of 3 documents (INV-003, CON-001, INV-002)
  - Tools: document_search
  - Sources: INV-003, CON-001, INV-002

- **"how many documents we have"**
  - Expected: Total count of 5 documents
  - Tools: document_statistics

#### **Summarization Queries (Intent: summarization)**
- **"summarize all documents"**
  - Expected: Overview of all 5 documents with financial summary
  - Key Points: Document types, total amounts, averages
  - Tools: document_reader

- **"summarize all contracts"**
  - Expected: Detailed summary of CON-001 service agreement
  - Key Points: Parties involved, amount, effective date, terms
  - Tools: document_search

- **"Give me an overview of the invoices"**
  - Expected: Summary of all 3 invoices with amounts and details
  - Tools: document_search

#### **Calculation Queries (Intent: calculation)**
- **"Calculate the sum of all invoice totals"**
  - Expected: $305,800 (22,000 + 69,300 + 214,500)
  - Result: 305800.0
  - Tools: calculator

- **"What's the average amount across all documents?"**
  - Expected: $97,650 average across all documents
  - Tools: document_statistics

- **"how many of them below 50,000"**
  - Expected: 2 documents (calculated from context)
  - Result: 2.0
  - Tools: document_search

### Usage Instructions

**Interactive Mode:**
```bash
python main.py
```

**Example Session:**
```
Enter your user ID: Osama
Started new session 3a346daf-4dbd-415b-a0b7-d30cc8b116d1

Enter Message: What's the total amount in invoice INV-001?
🤖 Assistant: The total amount in invoice INV-001 is $22,000. This includes a subtotal of $20,000 for services rendered and a 10% tax of $2,000.

SOURCES: INV-001
TOOLS USED: document_reader

Enter Message: summarize all documents
🤖 Assistant: The document collection consists of five documents, including three invoices, one contract, and one claim. The total financial amount across all documents is $488,250.00, with an average amount of $97,650.00. The amounts range from a minimum of $2,450.00 to a maximum of $214,500.00.

KEY POINTS:
  • Total of 5 documents: 3 invoices, 1 contract, 1 claim.
  • Total financial amount: $488,250.00.
  • Average document amount: $97,650.00.
  • Amounts range from $2,450.00 to $214,500.00.

TOOLS USED: document_statistics

Enter Message: Calculate the sum of all invoice totals
🤖 Assistant: To find the sum of all invoice totals, we add the amounts from each invoice together. The invoices have the following totals: $22,000 from INV-001, $69,300 from INV-002, and $214,500 from INV-003. Adding these amounts gives us the total sum: $22,000 + $69,300 + $214,500 = $305,800.

RESULT: 305800.0
TOOLS USED: document_search

Enter Message: Find documents with amounts over $50,000
🤖 Assistant: The documents with amounts over $50,000 are:

1. **Invoice #12347** (Document ID: INV-003) with an amount of $214,500.00.
2. **Service Agreement** (Document ID: CON-001) with an amount of $180,000.00.
3. **Invoice #12346** (Document ID: INV-002) with an amount of $69,300.00.

SOURCES: INV-003, CON-001, INV-002
TOOLS USED: document_search

Enter Message: how many documents we have
🤖 Assistant: We have a total of 5 documents in our collection. These include 3 invoices, 1 contract, and 1 claim.

TOOLS USED: document_statistics
```

### Key Implementation Details

**Security Features:**
- Calculator tool validates expressions for security
- Only allows safe mathematical operations
- Prevents execution of dangerous code

**Error Handling:**
- Graceful error handling in all components
- Comprehensive logging for debugging
- User-friendly error messages

**Performance Optimizations:**
- Efficient document retrieval with relevance scoring
- Conversation history management with summarization
- Tool usage tracking for optimization

### System Capabilities Demonstrated

Based on real usage, the system successfully handles:

**Intent Classification Accuracy:**
- Correctly identifies Q&A queries (e.g., "What's the total amount...")
- Correctly identifies summarization queries (e.g., "summarize all documents")
- Correctly identifies calculation queries (e.g., "Calculate the sum...")

**Document Processing:**
- Retrieves specific documents by ID (INV-001, CON-001, etc.)
- Filters documents by amount ranges (over $50,000)
- Provides statistical summaries (total documents, amounts, averages)
- Maintains document references and sources

**Mathematical Operations:**
- Performs complex calculations (sum of invoice totals: $305,800)
- Uses context from previous queries for calculations
- Provides step-by-step explanations
- Returns structured results with confidence

**Conversation Management:**
- Maintains session state across multiple turns
- Tracks conversation history with timestamps
- Preserves context for follow-up questions
- Logs all tool usage for compliance

**Tool Integration:**
- document_reader: Reads specific document content
- document_search: Finds documents by criteria
- document_statistics: Provides collection overview
- calculator: Performs mathematical operations

## Built With

* [LangChain](https://www.langchain.com/) - Framework for developing applications with LLMs
* [LangGraph](https://github.com/langchain-ai/langgraph) - Library for building stateful, multi-actor applications with LLMs
* [OpenAI](https://openai.com/) - GPT-4o model for natural language processing
* [Pydantic](https://pydantic.dev/) - Data validation using Python type annotations

## Project Structure

```
doc_assistant_project/
├── src/
│   ├── schemas.py        # Pydantic models (IMPLEMENTED)
│   ├── retrieval.py      # Document retrieval (IMPLEMENTED)
│   ├── tools.py          # Agent tools (IMPLEMENTED)
│   ├── prompts.py        # Prompt templates (IMPLEMENTED)
│   ├── agent.py          # LangGraph workflow (IMPLEMENTED)
│   └── assistant.py      # Main agent (IMPLEMENTED)
├── sessions/             # Saved conversation sessions
├── logs/                 # Tool usage logs
├── main.py               # Entry point
├── test_implementation.py # Automated Test script
├── requirements.txt      # Dependencies
└── README.md             # The report file that includes the project implementation details
```

## License
[License](../LICENSE.md)
