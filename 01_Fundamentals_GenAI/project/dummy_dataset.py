import pathlib
from typing import List
import pandas as pd


news_data = [
    {
        "Headline": "President Signs New Climate Accord",
        "Summary": "The president has committed to new international environmental targets aimed at reducing emissions by 2030.",
        "Topic": "politics",
    },
    {
        "Headline": "Markets Rally After Interest Rate Cut",
        "Summary": "Stocks soared as the central bank announced a reduction in interest rates to stimulate economic growth.",
        "Topic": "economy",
    },
    {
        "Headline": "Local Team Wins National Championship",
        "Summary": "Fans celebrated as the city's football team clinched the national title in a thrilling final match.",
        "Topic": "sports",
    },
    {
        "Headline": "New Bill Aims to Support Low-Income Families",
        "Summary": "Lawmakers are proposing a bill to increase financial aid and social services for struggling households.",
        "Topic": "society",
    },
    {
        "Headline": "New Breakthrough in Cancer Research",
        "Summary": "Scientists have developed a treatment that significantly improves recovery rates for certain cancers.",
        "Topic": "health",
    },
    {
        "Headline": "AI Startup Releases Groundbreaking Model",
        "Summary": "A tech company has unveiled a large language model that promises more accurate text generation.",
        "Topic": "technology",
    },
    {
        "Headline": "Blockbuster Film Tops Box Office",
        "Summary": "The new action-packed thriller has grossed over $200 million worldwide in its first week.",
        "Topic": "entertainment",
    },
    {
        "Headline": "Tensions Rise in Diplomatic Standoff",
        "Summary": "Leaders of two countries exchanged sharp rhetoric over disputed borders and trade sanctions.",
        "Topic": "international",
    },
    {
        "Headline": "Parliament Debates Election Reform",
        "Summary": "A heated discussion took place over changes to the voting system ahead of next year's elections.",
        "Topic": "politics",
    },
    {
        "Headline": "Unemployment Drops to Record Low",
        "Summary": "New reports show a significant decrease in jobless claims, indicating a recovering job market.",
        "Topic": "economy",
    },
    {
        "Headline": "Olympic Committee Announces New Host City",
        "Summary": "The next Summer Games will be held in a city known for its sporting history and infrastructure.",
        "Topic": "sports",
    },
    {
        "Headline": "Protests Erupt Over Housing Prices",
        "Summary": "Citizens marched through the capital demanding more affordable housing and rent control.",
        "Topic": "society",
    },
    {
        "Headline": "Researchers Discover Gene Linked to Alzheimer's",
        "Summary": "A genetic mutation believed to influence the onset of Alzheimer’s has been identified by researchers.",
        "Topic": "health",
    },
    {
        "Headline": "Major Software Update Released",
        "Summary": "The update includes several new features and security patches for the operating system.",
        "Topic": "technology",
    },
    {
        "Headline": "Music Awards Celebrate Emerging Artists",
        "Summary": "The annual ceremony highlighted up-and-coming musicians from around the world.",
        "Topic": "entertainment",
    },
    {
        "Headline": "UN Convenes Emergency Session on Conflict",
        "Summary": "The United Nations held an emergency meeting to address rising violence in a conflict zone.",
        "Topic": "international",
    },
    {
        "Headline": "President Faces Ethics Investigation",
        "Summary": "An independent commission is reviewing allegations of misconduct during the president’s previous term.",
        "Topic": "politics",
    },
    {
        "Headline": "Inflation Rates Soar Amid Global Uncertainty",
        "Summary": "Consumers are facing higher prices for essentials as inflation hits a decade-high.",
        "Topic": "economy",
    },
    {
        "Headline": "Champion Boxer Announces Retirement",
        "Summary": "After years of dominating the ring, the titleholder says it’s time to hang up the gloves.",
        "Topic": "sports",
    },
    {
        "Headline": "Nonprofits Launch Mental Health Campaign",
        "Summary": "A coalition of organizations is pushing for more awareness and resources for mental well-being.",
        "Topic": "health",
    },
]


def get_dummy_news_dataset_list() -> List:
    return news_data


def get_dummy_news_dataset_filepath(filepath: str | pathlib.Path = "dummy_news.csv") -> str:
    filepath = pathlib.Path(filepath)
    news_data = get_dummy_news_dataset_list()
    df = pd.DataFrame(news_data)
    df.to_csv(filepath.as_posix(), index=False)

    return filepath.as_posix()
