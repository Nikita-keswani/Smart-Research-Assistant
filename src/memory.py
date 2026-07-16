from langchain_classic.memory import ConversationBufferMemory as LangChainConversationBufferMemory
from config import config

class ConversationBufferMemory:

    def __init__(self , config = config):
        self.config = config
        self.build()
    
    def build(self) -> LangChainConversationBufferMemory:
        """Build a new memory from scratch."""
        print("[Memory] Initializing new conversation memory buffer...")
        self.memory = LangChainConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        return self.memory

    def reset(self) -> LangChainConversationBufferMemory:
        """Reset the memory."""
        print("[Memory] Clearing and resetting conversation memory buffer...")
        self.memory = self.build()
        return self.memory

    




        