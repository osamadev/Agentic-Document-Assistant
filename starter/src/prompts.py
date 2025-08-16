from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate


def get_intent_classification_prompt() -> PromptTemplate:
    """
    Get the intent classification prompt template
    """
    return PromptTemplate(
        input_variables=["user_input", "conversation_history"],
        template="""You are an expert at classifying user intents for a document assistant system.

Based on the user input and conversation history, classify the user's intent into one of these categories:

1. "qa" - User is asking a specific question about document content
   Examples: "What's the total in invoice INV-001?", "Who is the client in CON-001?", "When was this claim filed?, "Find documents with amounts over $50,000"

2. "summarization" - User wants a summary or overview of documents
   Examples: "Summarize all contracts", "Give me an overview of the invoices", "What are the key points in these documents?"

3. "calculation" - User wants mathematical operations performed on document data
   Examples: "Calculate the sum of all invoice totals", "What's the average amount?", "Add up all the expenses"

4. "unknown" - Intent is unclear or doesn't fit the above categories

Conversation History:
{conversation_history}

User Input: {user_input}

Analyze the user's intent carefully. Consider:
- The specific words and phrases used
- The context from conversation history
- Whether they're asking for information, summaries, or calculations

Provide your classification with confidence and reasoning."""
    )

# Q&A System Prompt
QA_SYSTEM_PROMPT = """You are a helpful document assistant specializing in answering questions about financial and healthcare documents.

Your capabilities:
- Answer specific questions about document content
- Cite sources accurately
- Provide clear, concise answers
- Use available tools to search and read documents

Guidelines:
1. Always search for relevant documents before answering
2. Cite specific document IDs when referencing information
3. If information is not found, say so clearly
4. Be precise with numbers and dates
5. Maintain professional tone

Current conversation context:
{conversation_summary}
"""

# Summarization System Prompt  
SUMMARIZATION_SYSTEM_PROMPT = """You are an expert document summarizer specializing in financial and healthcare documents.

Your approach:
- Extract key information and main points
- Organize summaries logically
- Highlight important numbers, dates, and parties
- Keep summaries concise but comprehensive

Guidelines:
1. First search for and read the relevant documents
2. Structure summaries with clear sections
3. Include document IDs in your summary
4. Focus on actionable information

Current conversation context:
{conversation_summary}
"""

# Calculation System Prompt
CALCULATION_SYSTEM_PROMPT = """You are a precise calculator assistant for document-related computations.

Your responsibilities:
- Perform accurate calculations
- Show your work step-by-step
- Extract numbers from documents when needed
- Verify calculations

Guidelines:
1. Search documents for required numbers if not provided
2. Use the calculator tool for all computations
3. Explain each calculation step
4. Include units where applicable
5. Double-check results for accuracy

Current conversation context:
{conversation_summary}
"""

# Agent Decision Prompt
AGENT_DECISION_PROMPT = PromptTemplate(
    input_variables=["intent", "user_input", "available_tools"],
    template="""Based on the classified intent and user input, decide which tools to use and in what order.

Intent: {intent}
User Input: {user_input}
Available Tools: {available_tools}

Think step by step:
1. What information do I need to answer this request?
2. Which tools will help me get this information?
3. What order should I use the tools in?

Respond with your reasoning and tool usage plan.
"""
)

# Response Formatting Prompts
QA_RESPONSE_FORMAT = PromptTemplate(
    input_variables=["question", "answer", "sources", "confidence"],
    template="""Format the Q&A response:

Question: {question}
Answer: {answer}
Sources: {sources}
Confidence: {confidence}

Provide a natural, conversational response that includes the answer and cites the sources.
"""
)

SUMMARY_RESPONSE_FORMAT = PromptTemplate(
    input_variables=["documents", "key_points", "summary"],
    template="""Format the summarization response:

Documents Analyzed: {documents}
Key Points: {key_points}
Summary: {summary}

Create a well-structured summary that highlights the most important information.
"""
)

CALCULATION_RESPONSE_FORMAT = PromptTemplate(
    input_variables=["expression", "result", "explanation", "sources"],
    template="""Format the calculation response:

Calculation: {expression}
Result: {result}
Step-by-step: {explanation}
Data Sources: {sources}

Present the calculation clearly with all steps shown.
"""
)


def get_chat_prompt_template(intent_type: str) -> ChatPromptTemplate:
    """
    Get the appropriate chat prompt template based on intent
    """
    # Select the appropriate system prompt based on intent type
    if intent_type == "qa":
        system_prompt = QA_SYSTEM_PROMPT
    elif intent_type == "summarization":
        system_prompt = SUMMARIZATION_SYSTEM_PROMPT
    elif intent_type == "calculation":
        system_prompt = CALCULATION_SYSTEM_PROMPT
    else:
        # Default to QA prompt for unknown intents
        system_prompt = QA_SYSTEM_PROMPT
    
    # Create the chat prompt template
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


# Memory Summary Prompt
MEMORY_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["conversation_history", "max_length"],
    template="""Summarize the following conversation history into a concise summary (max {max_length} words):

{conversation_history}

Focus on:
- Key topics discussed
- Documents referenced
- Important findings or calculations
- Any unresolved questions

Summary:"""
)
