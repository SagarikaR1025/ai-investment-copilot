from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def risk_analysis_agent(context, company):
    prompt = f"""
    You are a financial risk analyst.

    Analyze the following investment research context for {company}.

    Focus on:
    - financial risks
    - competitive threats
    - operational concerns
    - market uncertainty

    Context:
    {context}

    Provide a concise professional risk summary.
    """

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text


def investment_strategy_agent(context, company):
    prompt = f"""
    You are an investment strategy analyst.

    Analyze the following investment research context for {company}.

    Focus on:
    - growth opportunities
    - AI exposure
    - revenue drivers
    - strategic positioning

    Context:
    {context}

    Provide an executive-level investment outlook.
    """

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text
def executive_summary_agent(context, company):
    prompt = f"""
    You are an investment banking AI research assistant.

    Using the research context below for {company}, generate a concise executive summary.

    Include:
    - overall investment narrative
    - key growth drivers
    - major risks
    - one-line strategic takeaway

    Context:
    {context}

    Format the response with clear bullet points.
    """

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=450,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text
def final_recommendation_agent(context, company):
    prompt = f"""
    You are an AI investment research copilot for an investment banking team.

    Based on the research context below for {company}, provide a final recommendation-style summary.

    Include:
    - investment stance: Bullish, Neutral, or Cautious
    - top 2 reasons supporting the stance
    - top 2 risks
    - one recommended next research step

    Context:
    {context}

    Keep the response concise and executive-ready.
    """

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=450,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text