from agent_instructions import ReviewAgentInstructions, ScorerAgentInstructions
from agent_output import CommentOutput, PRComment
from clients.mcp_client import McpClientStreamableHttp
from clients.chat_client import ChatClient
from rag_agent import RAGAgent
from typing import Any, cast
import json
import numpy
import traceback

class ReviewScorer:
    def __init__(self, agentUrl: str, agentKey: str, agentModel: str, embeddingModel: str, embeddingDimension: int, ragPath:str = None):
        self.embeddingModel = embeddingModel
        self.embeddingDimension = embeddingDimension
        self.ragPath = ragPath

        self.mcpClient = McpClientStreamableHttp()
        self.chatClient = ChatClient(agentUrl, agentKey, agentModel, [], self.__tool_callback)

        self.__tools = []

        self.ragAgent = RAGAgent(agentUrl, agentKey, embeddingModel, embeddingDimension)
        try:
            self.ragAgent.load_index_data(ragPath, CommentOutput)
        except Exception as e:
            print(f"Error when loading initial RAG history: {str(e)}")
            traceback.print_exc()

    async def cleanup(self):
        self.ragAgent.save_index_data(self.ragPath)
        await self.ragAgent.cleanup()
        await self.mcpClient.cleanup()
        await self.chatClient.cleanup()

    async def connect_to_mcp_client(self, gitUrl: str, gitKey: str, ):
        await self.mcpClient.connect_to_streamable_http_server(gitUrl, {"Authorization": "Bearer " + gitKey})

        tools = await self.mcpClient.list_tools()
        self.__tools.clear()
        self.__tools.extend(tools)
        print("\nConnected to server with tools:", [tool["function"]["name"] for tool in self.__tools])

    async def score_pr_comment(self, prLink: str) -> Any:
        await self.__review_initial_pr(prLink)
        startingQuery = await self.__get_rag_query(prLink)
        print(f"Starting RAG: {startingQuery}")
        instructionTuple = ScorerAgentInstructions.ending_instruction({"url": prLink})
        query = startingQuery + instructionTuple[0]
        chatResult = await self.chatClient.process_query(query, self.__tools, instructionTuple[1])
        print(f"Score Result: {chatResult}")
        endingResult = cast(list[CommentOutput], chatResult)

        endingEmbeddings:numpy.ndarray = await self.ragAgent.get_embeddings([item.comment for item in endingResult])
        self.ragAgent.add_to_index(endingEmbeddings, endingResult)
        return None


    async def __review_initial_pr(self, prLink: str):
        for instruction in ReviewAgentInstructions.instructions_list({"url": prLink}):
            chatResult = await self.chatClient.process_query(instruction, self.__tools) 
            print(f"Instruction Result: {chatResult}")

        instructionTuple = ReviewAgentInstructions.ending_instruction({"url": prLink})
        chatResult = await self.chatClient.process_query(instructionTuple[0], self.__tools, instructionTuple[1])
        print(f"Review Result: {chatResult}")

    async def __get_rag_query(self, prLink: str) -> str:
        if self.ragAgent.faiss_index.ntotal == 0:
            return ""
        
        startingQuery, startingType = ScorerAgentInstructions.starting_instruction({"url": prLink})
        startingResult = cast(list[PRComment], await self.chatClient.process_query(startingQuery, self.__tools, startingType))
        startingEmbeddings:numpy.ndarray = await self.ragAgent.get_embeddings([item.comment for item in startingResult])
        similarScorings = cast(list[CommentOutput], self.ragAgent.retrieve_similar(startingEmbeddings))
        context_text = json.dumps(similarScorings)

        return f"Given these past PR comments with their metadata and scores:\n{context_text}\n"


    async def __tool_callback(self, name: str, args: dict, tool_call_id):
        """
        Callback from the chat client to execute a tool.
        TODO If we move to supporting multiple MCP servers, we'll need a mapping of tools
            to pick the right MCP client.
        """
        return await self.mcpClient.execute_tool(name, args, tool_call_id)
