"""Make questions to your documents using langchain.
===============================================
"""

import argparse
import glob
import os
import random
from loguru import logger
from pprint import pprint
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
import pandas as pd

# Load environment variables and assert that the API key is set
load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "gpt-3.5-turbo")

PROMPT_ORGANS_STATUS = """
    Are the following organs annomalous? Based  the answer on the follwoing report:

    "{report}"

    {format_instructions}
    """


class AnomalousAbdominalOrganOrStructures(BaseModel):
    """Whether the patient has an anomalous abdominal organ or structure.

    This model stores in booleans wheter the field is present in the
    given abdominal CT report or not.
    """

    liver: bool = Field(False, description="Whether there is an anomalous liver.")
    gallbladder: bool = Field(
        False, description="Whether there is an anomalous gallbladder."
    )
    pancreas: bool = Field(False, description="Whether there is an anomalous pancreas.")
    spleen: bool = Field(False, description="Whether there is an anomalous spleen.")
    kidneys: bool = Field(False, description="Whether there is an anomalous kidneys.")


def load_report(report_path: str) -> str:
    """Load the report from the given path."""
    with open(report_path, "r") as f:
        report = f.read()
    return report


def find_report_files_by_pattern(directory: str, pattern: str = "*.txt") -> list[str]:
    """Find all the files that match the given pattern (wildcard)."""
    return glob.glob(os.path.join(directory, pattern))



def main(args):
    """Main function."""
    logger.info(f"Loading report files in {args.directory}...")
    report_files = find_report_files_by_pattern(directory=args.directory, pattern=args.pattern)
    if args.debug:
        report_files = random.sample(report_files, 1)
    logger.info(f"Found {len(report_files)} report files.")
    
    res = input("Continue? [y/n]")
    if res != "y":
        return

    # Define parser for the output of the model
    parser = PydanticOutputParser(pydantic_object=AnomalousAbdominalOrganOrStructures)

    # Define the chat model
    llm = ChatOpenAI(openai_api_base=OPEN_AI_API_KEY, model_name=OPEN_AI_MODEL)

    message = HumanMessagePromptTemplate.from_template(
        template=PROMPT_ORGANS_STATUS,
    )

    chat_prompt = ChatPromptTemplate.from_messages([message])

    results = []

    for report_file in report_files:
        report = load_report(report_file)

        logger.info(f"Processing report using model {OPEN_AI_MODEL}...")
        chat_prompt_with_values = chat_prompt.format_prompt(
            report=report, format_instructions=parser.get_format_instructions()
        )

        output = llm(chat_prompt_with_values.to_messages())
        # Print result. Should look like this:
        # {'additional_kwargs': {},
        # 'content': '{"liver": false, "gallbladder": false, "pancreas": false, '
        #             '"spleen": false, "kidneys": false}',
        # 'example': False,
        # 'type': 'ai'}
        resutls = parser.parse(output.content)


        # Store the result in a dictionary
        result_dict = {
            "report_file": report_file,
            **resutls.dict(),
        }

        # Append the dictionary to the results list
        results.append(result_dict)

    # Create a dataframe from the results list
    df = pd.DataFrame(results)
    df.to_csv("results.csv")

    # Print the dataframe
    print(df)
    logger.success("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="Pattern to find the report files.",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Directory where the report files are located.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (use a single report).",
    )
    args = parser.parse_args()

    main(args)
