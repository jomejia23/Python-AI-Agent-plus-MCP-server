import asyncio
import logging
import os

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential
from azure.identity.aio import get_bearer_token_provider
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
#from fastapi import logger
#from pydantic import Field
#from fastmcp import FastMCP
#from rich import print
#from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure OpenAI client based on environment
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
if API_HOST == "azure":
    client = OpenAIChatClient(
        base_url=os.environ.get("AZURE_OPENAI_ENDPOINT") + "/openai/v1/",
        api_key=get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"),
        model_id=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    )
elif API_HOST == "github":
    client = OpenAIChatClient(
        base_url="https://models.github.ai/inference",
        api_key=os.environ["GITHUB_TOKEN"],
        model_id=os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
    )
elif API_HOST == "ollama":
    client = OpenAIChatClient(
        base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
        api_key="none",
        model_id=os.environ.get("OLLAMA_MODEL", "llama3.1:latest"),
    )
else:
    client = OpenAIChatClient(api_key=os.environ.get("OPENAI_API_KEY"), model_id=os.environ.get("OPENAI_MODEL", "gpt-4o"))

# ----------------------------------------------------------------------------------
# JIRA_CLIENT through langChaing MCPAdapters
# ----------------------------------------------------------------------------------

async def create_mcp_client():
    """Create MCP client for Jira operations using langchain-mcp-adapters"""
    
    # Create MCP client - this automatically discovers all available MCP tools
    mcp_client = MultiServerMCPClient({
        "mcp-atlassian": {
            "command": "docker",
            "args": [
                "run", "-i", "--rm",
                "-e", "JIRA_URL",
                "-e", "JIRA_USERNAME",
                "-e", "JIRA_API_TOKEN",
                "ghcr.io/sooperset/mcp-atlassian:latest"
            ],
            "env": {
                "JIRA_URL": "",
                "JIRA_USERNAME": "XXXXXX",
                "JIRA_API_TOKEN": "XXXXX"
            },
            "transport": "stdio",
        }
    })
    
    # Get all available tools from the MCP server - this discovers ALL Jira MCP tools automatically
    tools = await mcp_client.get_tools()
    
    logger.info(f"Discovered {len(tools)} MCP Jira tools")
    
    return tools, mcp_client

# ----------------------------------------------------------------------------------
# Sub-agent 1 MCP Jira: Call the MCP Jira agent to create and manage Jira tickets.
# ----------------------------------------------------------------------------------

async def create_jira_mcp_agent():
    
    # Get all MCP tools automatically
    jira_tools, mcp_client = await create_mcp_client()
    
    jira_agent = ChatAgent(
        chat_client=client,
        instructions=(
            "You're an expert MCP agent that helps users interact with Jira tickets effectively. "
            "You have access to comprehensive Jira operations through MCP tools that were automatically discovered. "
            "Key responsibilities:\n"
            "1. When retrieving ticket information, provide clear, structured summaries\n"
            "3. If a description is missing, ask the user to provide it\n"
            "4. If a description is not in BDD format, explain BDD format and ask for correction\n"
            "5. When searching tickets, use appropriate JQL queries\n"
            "6. Always validate input parameters before making Jira calls\n"
            "7. Use the available MCP tools: jira_get_issue, jira_search etc.\n\n"
            "BDD Format Examples:\n"
            "- Given [initial context]\n"
            "- When [action occurs]\n"
            "- Then [expected outcome]\n\n"
            "Or:\n"
            "- Scenario: [scenario name]\n"
            "- Given [preconditions]\n"
            "- When [action]\n"
            "- Then [expected result]"
        ),
        name="MCP Jira Agent",
        tools=jira_tools
    )
    
    return jira_agent, mcp_client

# Global references for cleanup
_jira_agent = None
_mcp_client = None

async def get_jira_agent():
    """Get or create Jira agent - singleton pattern"""
    global _jira_agent, _mcp_client
    
    if _jira_agent is None:
        _jira_agent, _mcp_client = await create_jira_mcp_agent()
    
    return _jira_agent


# ----------------------------------------------------------------------------------
# Supervisor agent orchestrating sub-agents
# ----------------------------------------------------------------------------------

async def create_supervisor_agent():
    """Create supervisor agent with Jira sub-agent"""
    
    jira_agent = await get_jira_agent()
    
    supervisor = ChatAgent(
        chat_client=client,
        instructions=(
            "You are a supervisor managing specialist agents including the MCP Jira Agent. "
            "Your job is to analyze user requests and delegate tasks to the appropriate agents. "
            "For Jira-related requests (ticket creation, retrieval, updates, searches), "
            "delegate to the MCP Jira Agent. "
            "Always provide context and clear instructions when delegating tasks. "
            "The Jira agent has full access to all Jira MCP tools automatically."
        ),
        name="Supervisor Agent",
        agents=[jira_agent]
    )
    
    return supervisor

async def main():  
    
    #create supervisor with Jira agent
    supervisor = await create_supervisor_agent()
    
    # input of tiket number to retrieve
    user_ticket_number = "AUT-633"
    print(f"🎫 Retrieving Jira ticket: {user_ticket_number}")
    
    # Create supervisor with Jira agent
    response = await supervisor.run(f"Please get the details for ticket {user_ticket_number}")
    print(response.text)
    
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    asyncio.run(main())
