from openai import OpenAI
import os
def mixtral_87_api(query):

    info = [{"role": "user", "content": query}]
    openai_api_key = "EMPTY"
    openai_api_base = "your_api_port"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=6
    )
    try:
        chat_response = client.chat.completions.create(
            model="/model/Mixtral-8x7B-Instruct-v0.1",
            messages=info,
            stop=["<|im_end|>"],
            temperature=0,
        )
        return chat_response.choices[0].message.content
    except:
        return 'None'

def mistral_7B_api(query):

    info = [{"role": "user", "content": query}]
    openai_api_key = "EMPTY"
    openai_api_base = "your_api_port"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=5
    )
    try:
        chat_response = client.chat.completions.create(
        model="/model/Mistral-7B-Instruct-v0.2",
        messages=info,
        stop=["<|im_end|>"],
        temperature=0,
        )
        return chat_response.choices[0].message.content
    except:
        return 'None'

def process(resp):
    if '因此' in resp:
        resp = resp.split('因此')[-1]
    if '所以' in resp:
        resp = resp.split('所以')[-1]
    if '是：' in resp:
        resp = resp.split('是：')[-1]
    if '返回' in resp:
        resp = resp.split('返回')[-1]
    resp = resp.replace('\n', '')
    return resp
def deepseek_33B_api(query):

    info = [{"role": "user", "content": query}]
    openai_api_key = "EMPTY"
    openai_api_base = "your_api_port"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=10
    )
    try:
        chat_response = client.chat.completions.create(
        model="/model/deepseek-coder-33b-instruct",
        messages=info,
        stop=["<|im_end|>"],
        temperature=0,
        )
        resp = chat_response.choices[0].message.content
        resp=process(resp)
        return resp
    except:
        return "None"

def qwen_14B_api(query):

    info = [{"role": "user", "content": query}]
    openai_api_key = "EMPTY"
    openai_api_base = "your_api_port"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=10
    )
    try:
        chat_response = client.chat.completions.create(
        model="/model/Qwen-14B-Chat",
        messages=info,
        stop=["<|im_end|>"],
        temperature=0,
        )
        return chat_response.choices[0].message.content
    except:
        return "None"

def chatgpt(query):
    os.environ["http_proxy"] = 'your_api_port'
    os.environ["https_proxy"] = 'your_api_port'
    info = [{"role": "user", "content": query}]
    openai_api_key = "your_key"
    client = OpenAI(
        api_key=openai_api_key,
        timeout = 6
    )
    try:
        resp = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=info,
            temperature=0.8,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
        return resp.choices[0].message.content
    except:
        return "None"

if __name__ == '__main__':
    # inst = '''Now we are doing the task of relation triple extraction. Given the text and the interested relation, we may also give the extracted content. Please identify the entities I want according to the given content.
    # The input I provide will contain the text information, the relation, the extracted results and the entity type I need. Please return the entities I want in the specified format.
    # The output content only contains the entities I want, and needs to match the relation and extracted content provided in the input. If there are multiple corresponding entities, please separate them with English commas. If there is no corresponding entity, return 'None'. The examples are as follows:
    # example1: INPUT: {'text':'Donations may be made to Special Olympics , Massachusetts , Attn : Donations , 450 Maple Street , Building 1 , Danvers , MA 01923 .','relation':'location contains','extracted content':{'subject':'Massachusetts'},'entity type I need':'object'}
    # OUTPUT: 'Danvers'
    # example2: INPUT: {'text':'Earlier in the day , Senator Jim Bunning , Republican of Kentucky and a Hall of Fame pitcher , testified that he thought that players who used steroids should have their records wiped from the book .','relation':'place lived','extracted content':{'object':'Kentucky'},'entity type I need':'subject'}
    # OUTPUT: 'Jim Bunning'
    # example3: INPUT: {'text':'Steve Jobs and Bill Gates founded Apple and Microsoft respectively.','relation':'founded','extracted content':{},'entity type I need':'subject'}
    # OUTPUT: 'Steve Jobs, Bill Gates'
    # example4: INPUT: {'text':'Steve Jobs and Bill Gates founded Apple and Microsoft respectively.','relation':'founded','extracted content':{'subject':'Steve Jobs'},'entity type I need':'object'}
    # OUTPUT: 'Apple'
    # example5: INPUT: {'text':'Ding Xuexiang graduated from the Department of Politics of Fudan University.','relation':'graduated from','extracted content':{'object':'Tsinghua University'},'entity type I need':'subject'}
    # OUTPUT: 'None'
    # Now, please complete the following task according to the requirements. Again, the output content only contains the entities I want, and needs to match the relation and extracted content provided in the input. If there are multiple corresponding entities, please separate them with English commas.
    # INPUT:'''
    # # inst += "{'text':'He 's got the tools , and he 's a great kid , but you ca n't expect him to be fielding like Omar Vizquel '' -- the Giants ' slick-fielding shortstop from Caracas .','relation':'place of birth','extracted content':{'subject':'Omar Vizquel'},'entity type I need':'object'}"
    # inst += "{'text':'He is , as Julie Gilhart , fashion director of Barneys New York , suggested , probably the only designer you could name who has 60-year-olds who think he is incredible and 17-year-olds who think he is way cool . ','relation':'person of company','extracted content':{'object':'Barneys New York'},'entity type I need':'subject'}"
    # inst += '\n' + 'OUTPUT:'
    inst = f'Predefine the following relationship list:["location contains","country of administrative divisions"], please extract all triples containing the above relationship from the following sentences.'+\
    """

Input:The top American commander for the Middle East said Thursday that the insurgency in Iraq had not diminished , seeming to contradict statements by Vice President Dick Cheney in recent days that the insurgents were in their '' last throes . 

Note that the relationship name of the triple must be selected from the above relationship list, and other relationships not listed are not considered. Please output according to the specified format below:
[{"em1Text": subject1, "em2Text": object1, "label": relationship1}, {"em1Text": subject2, "em2Text": object2, "label": relationship2}]
Note that the triple may not only have two, please imitate this format and output all triples that meet the requirements.
Here is an example:
Input: Homeowners in parts of Palm Beach County in Florida , for instance , must show that all doors and windows to habitable space are at least 10 feet from the generator 's exhaust outlet and that the sound level is no greater than 75 decibels at the property line , said Rebecca Caldwell , building official for the county .
Output: [{"em1Text": "Florida", "em2Text": "Palm Beach County", "label": "location contains"}]

Again, it is emphasized that the relationship of the triples output must be selected from the predefined list above, and no relationship not in the list can be output. At the same time, please output as many triples as possible that meet the requirements.
Please imitate this example, according to the input, output all triples containing the above relationship according to the format requirements. Note that when the entity (subject or object) can be split into two words (such as a comma or comma in the middle), it should be split into two triples instead of merging into one triple.
"""
    print(mixtral_87_api(inst))