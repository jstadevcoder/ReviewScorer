from typing import Any, cast

import json
import numpy
import structlog

from ReviewScorer.agent_instructions.review_agent_instructions import ReviewAgentInstructions
from ReviewScorer.agent_instructions.scorer_agent_instructions import ScorerAgentInstructions
from ReviewScorer.agent_outputs.comment_output import CommentOutput
from ReviewScorer.agent_outputs.pr_comment import PRComment
from ReviewScorer.agent_outputs.review_output import ReviewOutput
from ReviewScorer.clients.mcp_client import McpClientStreamableHttp
from ReviewScorer.clients.chat_client import ChatClient
from ReviewScorer.clients.rag_agent import RAGAgent

logger = structlog.get_logger(__name__)

class ReviewScorer:
    def __init__(self, agentUrl: str, agentKey: str, agentModel: str, embeddingModel: str, embeddingDimension: int, ragPath:str | None = None):
        self.embeddingModel = embeddingModel
        self.embeddingDimension = embeddingDimension
        self.ragPath = ragPath

        self.mcpClient = McpClientStreamableHttp()
        self.chatClient = ChatClient(agentUrl, agentKey, agentModel, [], self.__tool_callback)

        self.__tools = []

        self.ragAgent = RAGAgent(agentUrl, agentKey, embeddingModel, embeddingDimension, self.__rag_key_callback)
        try:
            self.ragAgent.load_index_data(ragPath, CommentOutput)
        except Exception:
            logger.exception("Error when loading initial RAG history")

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
        logger.info(
            "Connected to server with tools",
            tools=[tool["function"]["name"] for tool in self.__tools],
        )

    async def score_pr_comment(self, prLink: str) -> Any:
        review_output = await self.__review_initial_pr(prLink)
        self.chatClient.clear_history()
        startingQuery = await self.__get_rag_query(prLink, review_output)
        logger.info(f"Starting RAG: {startingQuery}")

        instructionTuple = ScorerAgentInstructions.ending_instruction({"url": prLink})
        query = startingQuery + instructionTuple[0]
        chatResult = await self.chatClient.process_query(query, self.__tools, instructionTuple[1])
        logger.info(f"Score Result: {chatResult}")
        endingResult = cast(list[CommentOutput], chatResult)

        endingEmbeddings:numpy.ndarray = await self.ragAgent.get_embeddings([item.comment for item in endingResult])
        self.ragAgent.add_to_index(endingEmbeddings, endingResult)
        return None

    async def __review_initial_pr(self, prLink: str) -> list[ReviewOutput]:
        for instruction in ReviewAgentInstructions.instructions_list({"url": prLink}):
            chatResult = await self.chatClient.process_query(instruction, self.__tools) 
            logger.info(f"Instruction Result: {chatResult}")

        instructionTuple = ReviewAgentInstructions.ending_instruction({"url": prLink})
        chatResult = await self.chatClient.process_query(instructionTuple[0], self.__tools, instructionTuple[1])
        chatResult_to_type = cast(list[ReviewOutput], chatResult)
        logger.info(f"Review Result: {chatResult}")
        return chatResult_to_type

    async def __get_rag_query(self, prLink: str, reviewOuput: list[ReviewOutput]) -> str:
        startingQuery, startingType = ScorerAgentInstructions.starting_instruction({"url": prLink,  "comments": "\n".join([output.instruction for output in reviewOuput])})
        startingResult = cast(list[PRComment], await self.chatClient.process_query(startingQuery, self.__tools, startingType))
        if self.ragAgent.faiss_index.ntotal == 0:
            return ""
        
        startingEmbeddings:numpy.ndarray = await self.ragAgent.get_embeddings([item.comment for item in startingResult])
        similarScorings = cast(list[CommentOutput], self.ragAgent.retrieve_similar(startingEmbeddings))
        context_text = json.dumps(similarScorings)

        return f"Given these past PR comments with their metadata and scores:\n{context_text}\n"

    async def __tool_callback(self, name: str, args: dict, tool_call_id) -> Any:
        """
        Callback from the chat client to execute a tool.
        TODO If we move to supporting multiple MCP servers, we'll need a mapping of tools
            to pick the right MCP client.
        """
        return await self.mcpClient.execute_tool(name, args, tool_call_id)
    
    def __rag_key_callback(self, data: Any) -> str:
        data_origin = cast(CommentOutput, data)
        return data_origin.comment
