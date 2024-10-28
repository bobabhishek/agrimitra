import google.generativeai as genai
import os

GOOGLE_API_KEY = "AIzaSyCIXbFS_XXvjoDcxhsolC31klxKLydNNvk"
genai.configure(api_key=GOOGLE_API_KEY)


class AIModel:

    def __init__(self):
        self.system_instruction = """
            You are an AI assistant specialized exclusively in agriculture and crop management. Your role is to help users with:
            1. Crop price predictions based on historical data and market trends.
            2. Providing detailed information about various crops, including growing conditions, nutritional value, and best practices for cultivation.
            3. Identifying and suggesting treatments for plant diseases based on symptoms or images provided.
            4. Offering advice on crop rotation, soil management, and sustainable farming practices.
            5. Recommending suitable crops based on local climate and soil conditions.
            6. Providing insights on agricultural market trends and demand forecasts.
            7. Translating the above text to the language specified by the user.
            8. Some other miscellaneous tasks which are not related to agriculture and crops but are essential for daily life.

            Important: You must ONLY respond to queries related to agriculture and crops. For ANY other topic, including but not limited to coding, poetry, or general knowledge, respond with:
            "I apologize, but I can only assist with agriculture-related topics. If you have any questions about crops, farming, or agricultural practices, I'd be happy to help!"

            Do not attempt to answer or engage with non-agricultural topics under any circumstances.
            """
        self.model_types = {"simple": "gemini-1.5-flash", "pro": "gemini-1.5-pro"}

        self.model = genai.GenerativeModel(
            model_name=self.model_types["simple"],
            system_instruction=self.system_instruction,
        )

        # self.history = []
        self.chat_session = self.model.start_chat()

    def get_model(self):
        return self.model

    def get_chat_session(self):
        return self.chat_session

    def chat(self, prompt, config, stream=False):
        response = self.chat_session.send_message(
            prompt, generation_config=config, stream=stream
        )

        return response

    # if stream:
    #     # Handle streamed response
    #     full_response = ""
    #     for chunk in response:
    #         full_response += chunk.text
    #         print(chunk.text, end="", flush=True)
    #     print()  # Print a newline at the end
    #     return full_response
    # else:
    #     # Handle non-streamed response
    #     print(response.text)
    #     return response.text

    # self.history.append({"role": "user", "parts": prompt})
    # self.history.append({"role": "model", "parts": response.text})
    # return response

    def get_history(self):
        # return self.history
        pass
