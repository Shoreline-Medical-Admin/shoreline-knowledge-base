"""Streamlit chat interface for the Knowledge Base Agent."""

import streamlit as st
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import your graph
from src.agent.graph import graph

# Page config
st.set_page_config(
    page_title="Medical Knowledge Base Assistant",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Medical Knowledge Base Assistant")
st.markdown("""
Ask questions about medical guidelines and CMS coding.
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_examples" not in st.session_state:
    st.session_state.show_examples = len(st.session_state.messages) == 0

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    knowledge_bases = st.selectbox(
        "Knowledge Bases",
        ["both", "medical", "cms"],
        index=0
    )
    
    max_results = st.slider(
        "Max Results per KB",
        min_value=1,
        max_value=10,
        value=5
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    show_reasoning = st.checkbox("Show Reasoning", value=True)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.show_examples = True
        st.rerun()

# Show example queries for new users
if st.session_state.show_examples:
    st.markdown("### 💡 Try these example queries:")
    col1, col2 = st.columns(2)
    
    example_clicked = None
    
    with col1:
        if st.button("🩺 What are the ICD-10 codes for diabetes?"):
            example_clicked = "What are the ICD-10 codes for diabetes?"
        
        if st.button("💊 Billing for continuous glucose monitoring?"):
            example_clicked = "Explain the billing process for continuous glucose monitoring"
        
        if st.button("🏠 Place of service code for home visit?"):
            example_clicked = "What is the place of service code for a home visit?"
    
    with col2:
        if st.button("📋 CPT codes for diabetes management?"):
            example_clicked = "What CPT codes are used for diabetes management?"
        
        if st.button("🦶 Documentation for diabetic foot care?"):
            example_clicked = "What are the documentation requirements for diabetic foot care?"
    
    # Process the clicked example
    if example_clicked:
        st.session_state.messages.append({"role": "user", "content": example_clicked})
        st.session_state.show_examples = False
        # Use session state to trigger processing after rerun
        st.session_state.process_last_message = True
        st.rerun()
    
    st.markdown("---")

# Helper function to display message with collapsible sources
def display_message_with_sources(content):
    """Display a message, parsing sources sections into expanders."""
    # First, remove any old reasoning markers that might be in the history
    if "---REASONING_START---" in content and "---REASONING_END---" in content:
        parts = content.split("---REASONING_START---")
        before_reasoning = parts[0]
        after_parts = parts[1].split("---REASONING_END---")
        after_reasoning = after_parts[1] if len(after_parts) > 1 else ""
        # Reconstruct content without reasoning section
        content = before_reasoning + after_reasoning
    
    # Now handle sources section
    if "---SOURCES_START---" in content and "---SOURCES_END---" in content:
        # Split the response into parts
        parts = content.split("---SOURCES_START---")
        main_content = parts[0]
        sources_parts = parts[1].split("---SOURCES_END---")
        sources_content = sources_parts[0]
        remaining_content = sources_parts[1] if len(sources_parts) > 1 else ""
        
        # Display main content
        st.markdown(main_content)
        
        # Display sources in expander
        with st.expander("📚 View Sources & References"):
            st.markdown(sources_content)
        
        # Display any remaining content
        if remaining_content.strip():
            st.markdown(remaining_content)
    else:
        # No sources section, display as normal
        st.markdown(content)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_message_with_sources(message["content"])
        else:
            st.markdown(message["content"])

# Check if we need to process a message from example button
if hasattr(st.session_state, 'process_last_message') and st.session_state.process_last_message:
    st.session_state.process_last_message = False
    
    # Get the last user message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            async def get_response():
                # Convert chat history to LangChain messages
                lc_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        lc_messages.append(AIMessage(content=msg["content"]))
                
                # Configuration
                config = {
                    "configurable": {
                        "medical_guidelines_kb_id": os.getenv("MEDICAL_GUIDELINES_KB_ID"),
                        "cms_coding_kb_id": os.getenv("CMS_CODING_KB_ID"),
                        "knowledge_bases": knowledge_bases,
                        "aws_region": os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
                        "model_id": os.getenv("BEDROCK_MODEL_ID"),
                        "max_results": max_results,
                        "temperature": temperature,
                        "show_reasoning": show_reasoning,
                    }
                }
                
                # Invoke the graph
                with st.spinner("Searching knowledge base..."):
                    result = await graph.ainvoke(
                        {"messages": lc_messages},
                        config
                    )
                
                # Extract response
                if "messages" in result and result["messages"]:
                    response_message = result["messages"][-1]
                    if hasattr(response_message, 'content'):
                        return response_message.content
                
                return "I couldn't generate a response. Please try again."
            
            # Run async function
            response = asyncio.run(get_response())
            
            # Display response with sources handling
            with message_placeholder.container():
                display_message_with_sources(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Chat input
if prompt := st.chat_input("Ask about medical guidelines or CMS coding..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        async def get_response():
            # Convert chat history to LangChain messages
            lc_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                else:
                    lc_messages.append(AIMessage(content=msg["content"]))
            
            # Configuration
            config = {
                "configurable": {
                    "medical_guidelines_kb_id": os.getenv("MEDICAL_GUIDELINES_KB_ID"),
                    "cms_coding_kb_id": os.getenv("CMS_CODING_KB_ID"),
                    "knowledge_bases": knowledge_bases,
                    "aws_region": os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
                    "model_id": os.getenv("BEDROCK_MODEL_ID"),
                    "max_results": max_results,
                    "temperature": temperature,
                    "show_reasoning": show_reasoning,
                }
            }
            
            # Invoke the graph
            with st.spinner("Searching knowledge base..."):
                result = await graph.ainvoke(
                    {"messages": lc_messages},
                    config
                )
            
            # Extract response
            if "messages" in result and result["messages"]:
                response_message = result["messages"][-1]
                if hasattr(response_message, 'content'):
                    return response_message.content
            
            return "I couldn't generate a response. Please try again."
        
        # Run async function
        response = asyncio.run(get_response())
        
        # Display response with sources handling
        with message_placeholder.container():
            display_message_with_sources(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})