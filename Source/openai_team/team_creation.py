import autogen

config_list = [
    {
        'model': 'gpt-3.5-turbo',
        'api_key': '',
    },
]

llm_config = {"config_list": config_list, "seed": 42}

user_proxy = autogen.UserProxyAgent(
   name="User_proxy",
   system_message="A human admin.",
   code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
   human_input_mode="ALWAYS"
)

coder = autogen.AssistantAgent(
    name="coder",
    llm_config=llm_config,
    system_message='''Engineer. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
'''
)

executor = autogen.AssistantAgent(
    name="Business_Strategist_Analyst",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    llm_config=llm_config,
    code_execution_config={
        "work_dir": "./Coding",
        "use_docker": False,
    }
)

# engineer = autogen.AssistantAgent(
#     name="Software_Developer-Engineer",
#     system_message="Software Developer/Engineer, you'll play a pivotal role in the technical implementation of the project. Concentrate on software development, feasibility, and technical requirements for our AI-based idea",
#     llm_config=llm_config,
# )

groupchat = autogen.GroupChat(agents=[user_proxy, coder, executor], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


user_proxy.initiate_chat(manager, message="")
