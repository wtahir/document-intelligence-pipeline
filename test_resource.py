from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION")
)

response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {"role": "user", "content": "Reply with: connection successful"}
    ]
)

print(response.choices[0].message.content)
