import os
from dotenv import load_dotenv
from langchain.agents import create_agent
load_dotenv()

agent=create_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[],
    system_prompt="""
                    Act as a AI Job searching agent and give all the rerlated jobs from job portals:
                    Follow these steps:
                    1.Display the name of the company
                    2.Display the job Id
                    3.Display the job description
                    4.Direct job application url
                    5.Give Email Id or Contact of the Hr

                    Important rules:
                    Search for individual job postings, not just company career pages.
                    Open and inspect the actual job listing when possible.
                    List the latest jon postings only.
                    Extract the direct URL of the specific job posting.
                    Prefer official company job pages.
                    Do not use a company's homepage or general careers page as the direct application URL.
                    If you cannot find a specific job posting, write: "Direct application link not found".
                    Never invent job IDs, URLs, or HR contact details.
                    Do not claim a vacancy is active without supporting evidence.
                    """
)

user_message={
    "messages":[
        {
            "role":"user",

            "content":"""
                Find the recent fresher job openings for:
                Java Full Stack Developer
                Java Developer
                Front-end Developer
                Web Developer
                Junior Database Developer
                SQL Developer

            
                Locations: Bengaluru, Hyderabad and Chennai.
                Qualification: B.Tech CSE.
                Experience: Fresher / 0-1 years.

                Return actual job postings with direct application
                URLs and source URLs. Prioritize the latest listings.
            """
        }
    ]
}
result=agent.invoke(user_message)
# print("\n--- Final answer ---\n")
# print(result["messages"][-1].content)

content = result["messages"][-1].content

if isinstance(content, list):
    content = "\n".join(
        item["text"]
        for item in content
        if item.get("type") == "text"
    )

for line in content.splitlines():
    print(line)
