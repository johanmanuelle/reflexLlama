import reflex as rx
import ollama

class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


DEFAULT_CHATS = {
    "Intros": [],
}


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat = "Intros"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        #model = self.openai_process_question
        model= self.llamaAI_process_question

        async for value in model(question):
            yield value

    async def llamaAI_process_question(self, question: str):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """

        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        # Clear the input and start the processing.
        self.processing = True
        yield

        # Build the messages.
        messages = [
            {
                'role': 'user',
                'content': 'why the sky is blue?',
            }
        ]
        for qa in self.chats[self.current_chat]:
            messages.append({'role': 'user', 'content': qa.question})
            messages.append({'role': 'assistant', 'content': qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]

        #start New session in llama tinyllama
        
        stream= ollama.chat(
            model='tinyllama',
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            #print(chunk['message']['content'], end='', flush=True)
            answer_text = chunk['message']['content']
            if answer_text is not None:
                self.chats[self.current_chat][-1].answer += answer_text
            else:
                answer_text=""
                self.chats[self.current_chat][-1].answer += answer_text
            self.chats = self.chats
            yield

        # Toggle the processing flag.
        self.processing = False
