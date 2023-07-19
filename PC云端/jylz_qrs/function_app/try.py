""" ━━━━━━神兽出没━━━━━━ 
 　　　┏┓　　　┏┓ 
 　　┃　　　　　　　┃ 
 　　┃　　　━　　　┃ 
 　　┃　┳┛　┗┳　┃ 
 　　┃　　　　　　　┃ 
 　　┃　　　┻　　　┃ 
 　　┃　　　　　　　┃ 
 　　┗━┓　　　┏━┛Code is far away from bug with the animal rotecting 
 　　　　┃　　　┃ 神兽保佑,代码无bug 
 　　　　┃　　　┃ 
 　　　　┃　　　┗━━━┓ 
 　　　　┃　　　　　　　┣┓ 
 　　　　┃　　　　　　　┏┛ 
 　　  　┗┓┓┏━┳┓┏┛ 
　　　　　┃┫┫　┃┫┫ 
　　　　　┗┻┛　┗┻┛ 
"""
# import openai
#
# # 设置OpenAI API 密钥
# openai.api_key = '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'
#
# # 定义问题
# question = "你好，课堂专注度总分为5分，我现在的得分是4.75分，请你写一份报告评价我的课堂表现。"
#
# # 调用ChatGPT来生成答案
# response = openai.Completion.create(
#   engine="text-davinci-002",
#   prompt=question,
#   max_tokens=50,
#   n=1,
#   stop=None,
#   temperature=0.7,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )
#
# # 提取答案
# answer = response.choices[0].text.strip()
#
# # 打印答案
# print(answer)
import openai

# Apply the API key
openai.api_key = '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'

# Define the text prompt
prompt = "In a shocking turn of events, scientists have discovered that "

# Generate completions using the API
completions = openai.Completion.create(
    engine="text-davinci-001",
    prompt=prompt,
    max_tokens=100,
    n=1,
    stop=None,
    temperature=0.5,
)

# Extract the message from the API response
message = completions.choices[0].text
print(message)