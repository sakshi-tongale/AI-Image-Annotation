
import ollama

print("Testing Qwen3-VL...")

try:
    response = ollama.chat(
        model="qwen3-vl:4b",
        messages=[
            {
                "role": "user",
                "content": "What is image annotation? Answer in 2 sentences."
            }
        ],
        options={
            "num_predict": 60
        }
    )

    print("\nAI Response:")
    print(response.message.content)

except Exception as e:
    print("Error:", e)
