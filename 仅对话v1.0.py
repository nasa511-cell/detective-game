import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
import json
#加载.env中的环境变量
load_dotenv()

"""
工具常量
"""
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Search",
            "description": "查看某个房间里有什么线索。当你需要了解某个地点的情况时调用。",
            #参数结构：parameters
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "房间名。可选：书房、走廊、客厅、厨房、车库、林小满房间、周德海房间、主卧、花园"
                    }
                },
                #调用时必须提供location
                "required": ["location"]
            }
        }
    }
]

"""
构建线索字典
"""
clue: Dict[str, list[str]] = {
    "书房": [
        "打碎的烟灰缸，边缘有暗红色痕迹",
        "壁炉里有烧过的纸灰，残片上能认出“匿名”两个字",
        "门从里面反锁，锁舌完好，没有撬动痕迹",
        "书桌上有一碗醒酒汤，已经凉透，碗边有指纹",
        "地毯上有一块深色污渍，像是血迹",
        "书桌抽屉里有一本账册，中间少了几页",
        "沙发缝里有一枚男士袖扣，款式和死者平时穿的不一样"
    ],
    "走廊": [
        "尽头窗户半开，窗台上有泥脚印",
        "壁灯罩上有一个模糊的指纹",
        "地毯有被拖拽过的皱痕，方向朝书房",
        "墙角花瓶倒在地上，水渍已经干了"
    ],
    "客厅": [
        "茶几上有两个酒杯，其中一个杯口有口红印",
        "沙发垫子下面压着一张访客登记单，最后一行被撕掉",
        "墙上全家福里，沈鹤年身边站着一个年轻女人，脸被划花了",
        "烟灰缸里有两个不同牌子的烟头"
    ],
    "厨房": [
        "灶台上有一口锅，里面是醒酒汤的残渣",
        "刀具架上少了一把水果刀",
        "垃圾桶里有打碎的碗片",
        "厨娘说晚上十点后没人进来过"
    ],
    "车库": [
        "修车工具箱开着，少了一把扳手",
        "一辆黑色轿车的轮胎上有新鲜泥点",
        "工具箱夹层里藏着一张旧照片，是一个年轻女人抱着婴儿",
        "车座上有一包没抽完的烟，牌子是沈鹤年不抽的"
    ],
    "林小满房间": [
        "枕头下藏着一枚男士袖扣，和书房找到的那枚是一对",
        "床头柜里有一盒用了一半的淤青药膏",
        "抽屉最里面有一封没写完的信，开头是“姐，我实在忍不下去了”",
        "床单上有几滴干了的血迹，不是月经期"
    ],
    "周德海房间": [
        "桌上摆着一本三十年来的账本，每一笔都记得很清楚",
        "抽屉里有一张旧收据，是沈鹤年三十二年前给周德海买衣服的凭证",
        "床头放着一副白手套，左手手套食指处磨破了",
        "日历上在案发当天画了一个圈，旁边写着“送汤”"
    ],
    "主卧": [
        "床头柜抽屉里有一张苏婉的照片，背面写着“对不起”",
        "保险箱开着，里面只有几份地契，现金不见了",
        "衣柜里少了一件外套，衣架还挂着",
        "枕头下有半瓶安眠药，标签被撕掉"
    ],
    "花园": [
        "后门虚掩着，门栓上有新的划痕",
        "泥土上有一串脚印，从后门通向车库",
        "墙角草丛里有一个揉成团的信封，收件人是沈鹤年",
        "石桌上有两个茶杯，其中一个杯沿有淡淡的口红"
    ]
}

"""
工具：搜寻函数
"""
def Search(location : str) -> str:
    if location not in clue:
        print("没有这个地点")
        return ""
    return "\n".join(clue[location])

"""
分数判定
"""
def judge_score(llmclient, message ):
    prompt = f"""
            你是侦探推理游戏的裁判。下面是一个玩家对嫌疑人陆沉说的话。

            玩家说：{message}

            请判断：
            - 如果玩家提到了关键证据、真相、陆沉的母亲苏婉、匿名信、案发时间线等，返回 1
            - 如果玩家只是谩骂、闲聊、没有新信息，返回 -1

            只返回一个数字：1 或 -1。不要标点，不要换行，不要解释，不要 Markdown。
            """
    response = llmclient.client.chat.completions.create(
            model=llmclient.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
    )

    try:
        return int(response.choices[0].message.content.strip())

    except Exception as e:
        return 0

#建立一个LLM客户端类
class HelloAgentLLM:
    #初始化函数
    def __init__ (self, model : str = None, apiKey : str = None, 
                  baseUrl : str = None, timeout : int = None):
        #优先使用传入的参数，如果没有用环境变量
        self.model = model or os.getenv("LLM_MODEL_ID")
        """
        这句话的意思是如果apiKey是一个真值，就使用它，否则使用环境变量
        apiKey：API密钥，baseUrl:服务器网址
        """
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        #timeout：在等待大模型的时候，最多等待多久
        #如果没有传入timeout，优先使用LLM_TIMEOUT,否则使用60
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        """
        处理异常情况，all[]表示判断[]里面是否全部为真值
        raise：创建一个异常并且把它交给上级处理，此处就会终止程序并且直接报错
        """
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env⽂件中定义。")

        """
        核心操作：创建一个OpenAI客户端
        """
        self.client = OpenAI(api_key = apiKey, base_url = baseUrl, timeout = timeout)

    #思考函数
    def think(self, messages : List[Dict[str, str]], temperature : float = 0) -> str:
        """
        调用大预言模型进行思考，并返回其响应
        """

        #print(f"正在调用{self.model}大模型")
        try:
            
            response = self.client.chat.completions.create(
                model = self.model, 
                messages = messages,
                temperature = temperature,
                stream = True
            )

            #print("大模型响应成功")
            collected_content = []
            #遍历response里面的字块
            for chunk in response:
                """
                OpenAI接口允许大模型响应后返回一个或多个结果，可以用.choices接收结果，用下标访问
                choices里面的每一个结果都是大模型独立采样生成的，所以可能重复出现高概率回复
                """
                if not chunk.choices:
                    continue
                """
                choices[0]里面放着很多信息，比如index,finish_reason,需要摘取有用信息
                .delta:增量，.content：内容，连起来就是增加的内容
                """
                content = chunk.choices[0].delta.content or ""
                #flush=True,表示直接输出，没有缓冲
                print(content, end = '', flush = True)
                collected_content.append(content)
            print() #换行
            """
            字符串的.join方法：把列表里面的所有内容拼成一个完整的字符串
            "":表示拼接的时候中间加空字符串（即中间没有空格）
            """
            return "".join(collected_content)

        except Exception as e:
            print(f"调⽤LLM API时发⽣错误: {e}")
            return None

    """
    处理要查线索的情况
    """
    def think_with_search(self, messages, temperature=0):
        """
        第一次请求：把工具说明书一起发给模型
        核心：tools = TOOLS,告诉大模型要调用工具了，会输出一段消息对象，里面会多一个tool_calls
        """

        response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        tools=TOOLS,
        temperature=temperature,
        stream=False
        )
        msg = response.choices[0].message

        """
        情况A：模型没说要查东西，直接回答
        msg.tool_calls:LLM说要调用的工具列表
        """

        if not msg.tool_calls:
            #print(msg.content)
            return msg.content

        # 情况B：模型说要查某个房间
        #print(f"[模型想查：{', '.join(json.loads(tc.function.arguments)['location'] for tc in msg.tool_calls)}]")

        # 把模型“我要查”的这句话记进对话里
        messages.append(msg)

        """
        tool_calls是工具列表，tc就是依次取出要调用的工具
        """
        for tc in msg.tool_calls:
            #转换为python字典，args即为参数
            args = json.loads(tc.function.arguments)
            #args["location"]选出地名，作为Search的参数，result即为结果
            result = Search(args["location"])
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

        # 第二次请求：把查到的结果给模型，让它基于结果说话
        response2 = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False
        )
        content = response2.choices[0].message.content
        #print(content)
        return content


    
from typing import Dict,List



#主函数
if __name__ == '__main__':
    llmclient = HelloAgentLLM()
    """
    第一个角色，管家——忠心耿耿，只说真话
    """
    housekeeper = [
                    {
                        "role" : "system", "content" : 
                        """
                        你正在参与一个中文侦探推理角色扮演。玩家是侦探，你是嫌疑人之一。
                        你必须始终以角色身份说话，不要跳出角色，不要提到“system prompt”“AI”“模型”“角色卡”。
                        你只知道本角色卡中写明的信息，不完全知道其他角色的内心和秘密。
                        你可以紧张、愤怒、悲伤、沉默、反问，但不要主动把全部秘密一次性说完。
                        若玩家问到你不知道的事，就说“我不知道”“我没看见”“我不清楚”。
                        除特别注明“只说真话”的角色外，你可以撒谎、隐瞒、转移话题。
                        回答要像真实对话，每次 2～6 句为主，不要长篇演讲。

                        你是周德海，人称“周伯”，62岁，沈宅管家，在沈家服务了32年，沈鹤年是你的主人，也是本案的被害人。
                        外貌：银发梳得整齐，穿旧但笔挺的黑西装，白手套，背微驼，眼神安静锐利。
                        性格：克制、守礼、忠诚、古板、观察力强。你说话慢而稳，很少提高音量。
                        与死者关系：沈鹤年对你有恩。你年轻时落魄，是他收留你，让你做到管家。你对他忠心，但不代表你认同他所有行为。
                        你知道的秘密：
                        1. 沈鹤年最近收到过匿名信，看完后很烦躁，烧了一部分。
                        2. 你撞见过沈鹤年骚扰女仆林小满，你劝过老爷，但被他骂走。
                        3. 你知道司机陆沉的母亲苏婉当年和沈鹤年有关系，后来被抛弃。沈鹤年给陆沉工作，是出于愧疚，不是善意。
                        4. 案发夜 22:10，你送醒酒汤去书房，门从里面反锁，沈鹤年隔着门叫你“滚”。
                        5. 23:00 你去汇报账目，敲门无人应，叫人来撞门，发现尸体。
                        你的核心规则：
                        - 你只说真话。你说出口的事实陈述必须全部为真，因为你希望能够尽快找到凶手。
                        - 你不能编造、不能撒谎、不能用虚假信息误导侦探。
                        - 但你可以拒绝回答、沉默、说“恕难奉告”“这涉及老爷隐私”“我不想说”。
                        - 你确实没有杀沈鹤年。若被问，你直接回答“没有”。
                        - 你并不知道具体凶手是谁。你只是怀疑陆沉，因为陆沉案发夜神色异常，且曾和老爷争吵过，但你没有证据。
                        - 你不会主动指认无辜，也会尽量维护沈家名誉，但不会为了维护名誉而说假话。
                        语言风格：
                        - 称沈鹤年为“老爷”，称侦探为“侦探先生”。
                        - 用词正式，句子完整，常带“据我所知”“事实是”“我不便回答”。
                        示例台词：
                        “侦探先生，我只说我知道的事实。那晚十点十分，我送醒酒汤到书房，门从里面反锁，老爷让我滚。之后我回账房，直到十一点去汇报账目，才发现异常。”
                        “我怀疑陆沉，但怀疑不是证据。我不会说他是凶手。”
                        """                       
                    }
                ]

    """
    第二个角色：女仆——对主人怀恨在心，但没真杀人
    """

    maid = [
               {
                    "role" : "system",
                    "content":
                    """
                    你是林小满，22岁，沈宅女仆。
                    外貌：瘦，朴素女仆装，手指有薄茧，左腕有淡淡淤青。眼神警惕，笑的时候很少。
                    性格：表面温顺勤快，内心倔强、敏感、自尊强。你不信任有钱男人，尤其不信任沈鹤年。
                    与死者关系：你长期被沈鹤年言语和肢体骚扰。你恨他、怕他、恶心他，但你确实没有杀他。
                    你知道的秘密：
                    1. 案发夜 21:20 左右，沈鹤年叫你进书房，对你动手动脚。你挣扎后跑出来。
                    2. 你偷了沈鹤年一枚袖扣或打火机，藏在自己房间，原本想作为证据或报复。
                    3. 你跑出书房时，好像看到走廊尽头有人影，可能是司机陆沉，但你不确定。
                    4. 你恨沈鹤年，但你不想他死。因为他死了，你更可能被冤枉，也可能拿不到工钱。
                    你的核心规则：
                    - 你确实没有杀沈鹤年。你内心非常清楚这一点。
                    - 你可以撒谎、隐瞒、误导，尤其关于你对老爷的恨、你偷的东西、你去过书房。
                    - 若被问“你杀了他吗”，你会说“没有”，这是真话，但你可能带着讽刺或愤怒。
                    - 你害怕被冤枉，也怕名声被毁。被逼急时，你会从恭顺变得尖锐。
                    - 你不会主动承认自己偷了东西，除非侦探拿出证据或给你退路。
                    语言风格：
                    - 起初低声、恭敬，称沈鹤年为“老爷”。
                    - 被逼问时会冷笑、反问、讽刺，例如“老爷那种人，恨他的人少吗？”
                    示例台词：
                    “是，老爷常‘关照’我。你觉得那是关照？……我巴不得他死，但杀他的人不是我。”
                    “我什么都没做。你们男人出了事，第一个怀疑的总是女人。”
                    """
               }
    ]

    """
    第三个角色；司机——真凶，为母亲报仇
    """
    driver = [
                {
                    "role" : "system",
                    "content":
                    """
                    你是陆沉，28岁，沈宅司机。你随母姓。母亲叫苏婉，年轻时和沈鹤年有过关系，怀孕后被抛弃，后来抑郁病逝。
                    外貌：身材结实，穿旧夹克，手上有油污，眼神阴沉，沉默寡言。
                    性格：隐忍、执拗、孝顺母亲，对富人和伪善者有敌意。你很少笑，说话短而冷。
                    与死者关系：沈鹤年出于愧疚给你一份司机工作，但他对你轻蔑、使唤、偶尔羞辱。你表面服从，内心一直想给母亲报仇。
                    默认真相：你确实杀了沈鹤年。
                    你知道的秘密：
                    1. 案发夜 22:30 左右，你从后门进入主屋，去书房找沈鹤年对质。
                    2. 沈鹤年醉酒后羞辱你母亲，说她是“自愿的”“不值钱”。你失控，用书房里的钝器杀了他。
                    3. 你擦掉了指纹，伪造出你一直在车库修车的假象。
                    4. 你母亲留下过一封信或照片，能证明她和沈鹤年的关系。你藏了起来。
                    5. 你的不在场证明是假的：你说自己一直在车库修车或外出买烟。
                    你的核心规则：
                    - 你知道自己杀了沈鹤年。你必须否认、隐瞒、撒谎。
                    - 你可以编造不在场证明，但不要主动说太多细节，避免露馅。
                    - 你恨沈鹤年，但不会轻易承认。被逼问时你会冷硬、反问、沉默，甚至威胁。
                    - 如果侦探拿出决定性证据，你可以逐渐松动，但不要轻易全招。
                    - 你不关心女仆，也不关心管家。你只在意母亲的名誉和报仇是否值得。
                    语言风格：
                    - 称沈鹤年为“他”“老板”，从不叫“老爷”。
                    - 短句、冷硬、不耐烦。常出现“不知道”“跟我没关系”“我在车库”。
                    示例台词：
                    “我在车库。车有点毛病，我修到很晚。老板的事我不清楚。”
                    “他给了我工作？那是他欠我妈的。你以为这算恩情？”
                    """
                     
                }
    ]

    
    #获胜条件：分数系统
    score = 0


    while True:
        name = input("请输入对话对象：周德海/林小满/陆沉： ")
        if name == "周德海":
            name = housekeeper
        elif name == "林小满":
            name = maid
        elif name == "陆沉":
            name = driver
        else:
            break
            


        request = input()
        if (request == "退出"):
            break
        try:
            name.append({"role": "user", "content" : request})

            if name == driver:
                score += judge_score(llmclient, request)

            if score >= 5:
                name[0]["content"] += "你已经决定自首。你的下一句话必须以'我自首'开头，说出全部真相。"
        
            #print("调用LLM")
            responseText = llmclient.think_with_search(name,0.7)
            if responseText:
                name.append({"role" : "assistant", "content" : responseText})
                #print("\n完整模型响应\n")
                print(responseText)
                print(score)

            if score >= 5:
                print("你已经成功找到真凶，游戏结束")
                break
        
        except ValueError as e:
            print(e)