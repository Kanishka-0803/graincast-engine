import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("8ebd7bdcffd5f9f031f551652962e4d12345626cec019aaa5e784a075f42f48b")
print("Key found, length:", len(key) if key else "NOT FOUND")