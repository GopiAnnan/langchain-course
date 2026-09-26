from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def main() -> None:
    print("Hello from langchain-course!")
    information = """Arun Rupesh Maini[a] (born 24 October 1995), commonly known as Mrwhosetheboss, is an English YouTuber who is best known for his technology-related content, and is the creator of one of the biggest tech-related YouTube channels.

Early life
Arun Rupesh Maini[2][3] was born on 24 October 1995[4][5] in Nottingham, England[6][7] and is of Indian origin.[3] His mother is from India, and moved to the UK when she was 15.[8]: 1:18  Maini's father was born in England.[8]: 1:41  As a child, he attended two schools. On weekdays, he would attend a regular English school, and, on weekends, he would attend a Hindi school.[9] He was educated at the fee-paying Nottingham High School,[10] and then moved on to study economics at the University of Warwick in Coventry, England.[11]

Career
Maini originally began by creating video game related content. When he was 14 years old, Maini's brother gave him his first smartphone, a ZTE Blade. Arun "fell in love with it", making a video about the phone, which performed "much better than I expected". This caused Arun to turn his attention to creating videos about smartphones.[11][12]

During his time at university, Maini had an eight-week internship at Pricewaterhouse Coopers, an accounting firm in London. When he finished this internship, he was offered an entry-level job, which would earn a salary of $35,000. However, he turned down the job and opted to focus more on his YouTube career.[11]

The Mrwhosetheboss channel initially focused on videos about smartphones. However, as the channel gained a following, Maini extended his video topics to cover other types of technology, and has since also made other opinion-related technology videos.[11] In 2015, Maini uploaded his first viral video, which was a tutorial on creating a makeshift 3D hologram projector by crafting a pyramid composed of reflective material and placing it on a smartphone screen.[13][14]

In May 2021, Maini signed with Night Media.[15] In September 2022, Maini released a video claiming that Samsung phones may have problems with swelling batteries after he found that his Samsung Galaxy Note 8, Samsung Galaxy S6, and Samsung Galaxy S10 had all experienced the issue. Similar claims were corroborated in the video by YouTuber Marques Brownlee.[16][17]

In August 2024, after achieving a goal he set himself of overtaking Apple in YouTube subscribers, Maini, along with Matthew Perks, built a 2.054-metre (6 ft 8.9 in) replica of the iPhone 15 Pro Max, which, on 29 August, achieved a Guinness World Record for the largest smartphone replica.[18][19] In 2024, Maini uploaded a series on his YouTube channel showcasing the features of his newly purchased tech-enabled home.[20]

In April 2026, Maini was selected to play for the YouTube Allstars in the 2026 Sidemen Charity Match, in which he scored the winning penalty for the Allstars.[21]

Personal life
In April 2026, Maini announced that he and his wife, Dhrisha Mehta, were expecting their first child, due in August 2026.[22] On 15 August, Maini announced that their child had been born. https://www.instagram.com/p/DcV-Iv3jJj3/?img_index=11"""

    summary_template = PromptTemplate(
        input_variables=["information"],
        template="""given the information {information} about a person, I want you to create:
        1. a short summary
        2. a list of 5 interesting facts about the person"""
    )
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_template | llm
    result = chain.invoke(input={"information": information})
    print(result.content)

if __name__ == "__main__":
    main()