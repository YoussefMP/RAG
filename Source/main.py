from autogen import oai

# MODEL = "vicuna-7b-v1.3"
MODEL = "fastchat-t5-3b-v1.0"
IPV4 = "3.70.232.239"

# create a text completion request
# response = oai.Completion.create(
#     config_list=[
#         {
#             "model": "vicuna-7b-v1.3",
#             "api_base": f"http://{IPV4}:80/v1",
#             "api_type": "open_ai",
#             "api_key": "NULL",                      # just a placeholder
#         }
#     ],
#     prompt="Hi",
# )
# print(response)

# create a chat completion request
response = oai.ChatCompletion.create(
    config_list=[
        {
            "model": MODEL,
            "api_base":  f"http://{IPV4}:80/v1",
            "api_type": "open_ai",
            "api_key": "NULL",
        }
    ],
    messages=[{"role": "user", "content": "Am I overloading you?"}]
)

print(f"Model-2: \"{response.choices[0].content}\"")
