import os
import pandas as pd
from transformers import pipeline
from sheethelper import SheetHelper
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer


from gp_patient_survey.params import *
from gp_patient_survey.utils import *
from gp_patient_survey.automation.git_merge import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
init(autoreset=True)
warnings.filterwarnings("ignore")
secret_path = os.getenv("SECRET_PATH")
from sheethelper import *
import cronitor

cronitor.api_key = os.getenv("CRONITOR_API_KEY")
monitor = cronitor.Monitor("AsmpQK")


@time_it
def load_google_sheet():
    sh = SheetHelper(
        sheet_url="https://docs.google.com/spreadsheets/d/1mtbKRaMrqdkaj6GqwTVFdblpdBq5APU4RDpb4jzdchY/edit#gid=1285209116",
        sheet_id=0,
    )
    data = sh.gsheet_to_df()
    data.columns = [
        "time",
        "surgery",
        "phone",
        "appointment_time",
        "making_appointment",
        "overall_experience",
        "website",
        "free_text",
    ]
    data["time"] = pd.to_datetime(data["time"], format="%d/%m/%Y %H:%M:%S")

    return data


@time_it
def word_count(df):
    df["free_text_len"] = df["free_text"].str.split().apply(len)
    return df


@time_it
def convert_phone_responses(df, column_name):
    # Define the mapping from string responses to numeric values
    mapping = {
        "Very easy": 4,
        "Fairly easy": 3,
        "Not very easy": 2,
        "Not at all easy": 1,
    }
    # Apply the mapping to the specified column
    df[column_name] = df[column_name].map(mapping)
    return df


@time_it
def convert_appointment_time(df, column_name):
    # Define the mapping from string responses to numeric values
    mapping2 = {
        "Very satisfied": 5,
        "Fairly satisfied": 4,
        "Neither satisfied nor dissatisfied": 3,
        "Fairly dissatisfied": 2,
        "Very dissatisfied": 1,
    }
    # Apply the mapping to the specified column
    df[column_name] = df[column_name].map(mapping2)
    return df


@time_it
def convert_other(df, column_name):
    # Define the mapping from string responses to numeric values
    mapping3 = {
        "Very good": 5,
        "Fairly good": 4,
        "Neither good nor poor": 3,
        "Fairly poor": 2,
        "Very poor": 1,
    }
    # Apply the mapping to the specified column
    df[column_name] = df[column_name].map(mapping3)
    return df


# Load a pre-trained NER pipeline
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
)


# Function to anonymize names in text
@time_it
def anonymize_names_with_transformers(text):
    # Run the NER pipeline on the input text
    entities = ner_pipeline(text)
    anonymized_text = text
    # Iterate over detected entities
    for entity in entities:
        # Check if the entity is a person
        if entity["entity_group"] == "PER":
            # Replace the detected name with [PERSON]
            anonymized_text = anonymized_text.replace(entity["word"], "[PERSON]")
    return anonymized_text


@time_it
def add_row_values(df, columns):
    # Ensure columns are in the DataFrame
    columns_present = [col for col in columns if col in df.columns]

    # Calculate the sum of the specified columns row-wise
    df["row_sum"] = df[columns_present].sum(axis=1)

    return df


@time_it
def text_classification(data):
    # Initialize classifier
    classifier = pipeline(
        task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None
    )

    # Initialize lists to store labels and scores
    classif = []
    classif_scores = []

    # Iterate over DataFrame rows and classify text
    for _, row in data.iterrows():
        sentence = row["free_text"]
        model_outputs = classifier(sentence)
        classif.append(model_outputs[0][0]["label"])
        classif_scores.append(model_outputs[0][0]["score"])

    # Add labels and scores as new columns
    data["classif"] = classif
    data["classif_scores"] = classif_scores

    return data


# Function to replace surnames in text
def replace_surname(text, surnames_to_find):
    # Convert the text to lowercase for case-insensitive matching
    lower_text = text.lower()
    for surname in surnames_to_find:
        # Ensure the surname is also in lowercase
        lower_surname = surname.lower()
        # Replace the surname in the text with "🧍🏻", using the lowercase versions for comparison
        lower_text = lower_text.replace(lower_surname, "🧍🏻")
    # Return the modified text with surnames replaced
    return lower_text


@time_it
def anonymize(df, surnames_to_find):
    # Apply the function to the 'free_text' column with a lambda to ensure case-insensitive replacement
    df["free_text"] = df["free_text"].apply(
        lambda text: replace_surname(text, surnames_to_find)
    )
    return df


@time_it
def textblob_sentiment(data):
    data["free_text"] = data["free_text"].fillna("").astype(str)

    def analyze_sentiment(text):
        if text:
            sentiment = TextBlob(text).sentiment
            return pd.Series(
                [sentiment.polarity, sentiment.subjectivity],
                index=["polarity", "subjectivity"],
            )
        else:
            return pd.Series([0, 0], index=["polarity", "subjectivity"])

    sentiments = data["free_text"].apply(analyze_sentiment)
    data = pd.concat([data, sentiments], axis=1)

    # Check if the number of rows matches
    if len(sentiments) != len(data):
        raise ValueError("Mismatched row count between original data and sentiments")

    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Define the function to be applied to each row
    def get_sentiment(row):
        # Analyze sentiment using SentimentIntensityAnalyzer
        score = sia.polarity_scores(row["free_text"])

        # Assign the scores to the row
        for key in ["neg", "neu", "pos", "compound"]:
            row[key] = score[key]

        # Determine the overall sentiment based on the scores
        row["sentiment"] = "neutral"  # Default to neutral
        if score["neg"] > score["pos"]:
            row["sentiment"] = "negative"
        elif score["pos"] > score["neg"]:
            row["sentiment"] = "positive"

        return row

    # Apply the function to each row
    data = data.apply(get_sentiment, axis=1)

    return data


# Zer0-shot classification - do_better column
def batch_generator(data, column_name, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[column_name][
            i : i + batch_size
        ], i  # Yield the batch and the starting index


# Zero-Shot Classification (facebook model tried), now review the BartForConditionalGeneration
# facebook/bart-large-mnli
# trl-internal-testing/tiny-random-BartForConditionalGeneration ❌
# ybelkada/tiny-random-T5ForConditionalGeneration-calibrated
# MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
@time_it
def improvement_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    # Labels
    improvement_labels_list = [
        "Appointment Accessibility",
        "Reception Staff Interaction",
        "Medical Staff Competence",
        "Patient-Doctor Communication",
        "Follow-Up and Continuity of Care",
        "Facilities and Cleanliness",
        "Prescription and Medication Management",
        "Referral Efficiency",
        "Emergency Handling",
        "Patient Privacy and Confidentiality",
        "Telehealth Services",
        "Patient Education and Resources",
        "Waiting Room Comfort",
        "Patient Empowerment and Support",
        "Health Outcome Satisfaction",
        "Cultural Sensitivity",
        "Accessibility for Disabled Patients",
        "Mental Health Support",
        "Ambiance of Facility",
        "Nursing Quality",
        "Online Services & Digital Health",
        "Patient Safety",
        "Weekend Service Availability",
        "Telephone Service",
        "Overall Patient Satisfaction",
        "Blood Test Results & Imaging",
        "Patient Participation Group",
        "Doctor Consultations",
        "Home Visits",
        "Cancer Screening",
        "Vaccinations",
        "Test Results",
    ]

    # Initialize the list to store labels
    improvement_labels = [""] * len(data)  # Pre-fill with empty strings

    # Iterate over batches
    for batch, start_index in batch_generator(data, "do_better", batch_size):
        # Filter out empty or whitespace-only sentences
        valid_sentences = [
            (sentence, idx)
            for idx, sentence in enumerate(batch)
            if sentence and not sentence.isspace()
        ]
        sentences, valid_indices = (
            zip(*valid_sentences) if valid_sentences else ([], [])
        )

        # Classify the batch
        if sentences:
            model_outputs = classifier(
                list(sentences), improvement_labels_list, device="cpu"
            )
            # Assign labels to corresponding indices
            for output, idx in zip(model_outputs, valid_indices):
                improvement_labels[start_index + idx] = output["labels"][0]
                print(
                    f"{Fore.GREEN}Batch processed: {start_index + idx + 1}/{len(data)}"
                )

    # Add labels as a new column
    data["improvement_labels"] = improvement_labels
    return data


@time_it
def feedback_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    # Labels
    categories = [
        "Appointment Accessibility",
        "Reception Staff Interaction",
        "Medical Staff Competence",
        "Patient-Doctor Communication",
        "Follow-Up and Continuity of Care",
        "Facilities and Cleanliness",
        "Prescription and Medication Management",
        "Referral Efficiency",
        "Emergency Handling",
        "Patient Privacy and Confidentiality",
        "Telehealth Services",
        "Patient Education and Resources",
        "Waiting Room Comfort",
        "Patient Empowerment and Support",
        "Health Outcome Satisfaction",
        "Cultural Sensitivity",
        "Accessibility for Disabled Patients",
        "Mental Health Support",
        "Ambiance of Facility",
        "Nursing Quality",
        "Online Services & Digital Health",
        "Patient Safety",
        "Weekend Service Availability",
        "Telephone Service",
        "Overall Patient Satisfaction",
        "Blood Test Results & Imaging",
        "Patient Participation Group",
        "Doctor Consultations",
        "Home Visits",
        "Cancer Screening",
        "Vaccinations",
        "Test Results",
    ]

    # Initialize the list to store labels
    feedback_labels = [""] * len(data)  # Pre-fill with empty strings

    # Iterate over batches
    for batch, start_index in batch_generator(data, "free_text", batch_size):
        # Filter out empty or whitespace-only sentences
        valid_sentences = [
            (sentence, idx)
            for idx, sentence in enumerate(batch)
            if sentence and not sentence.isspace()
        ]
        sentences, valid_indices = (
            zip(*valid_sentences) if valid_sentences else ([], [])
        )

        # Classify the batch
        if sentences:
            model_outputs = classifier(list(sentences), categories, device="cpu")
            # Assign labels to corresponding indices
            for output, idx in zip(model_outputs, valid_indices):
                feedback_labels[start_index + idx] = output["labels"][0]
                print(
                    f"{Fore.GREEN}Batch processed: {start_index + idx + 1}/{len(data)}"
                )

    # Add labels as a new column
    data["feedback_labels"] = feedback_labels
    return data


@time_it
def concat_save_final_df(processed_df, new_df):
    combined_data = pd.concat([processed_df, new_df], ignore_index=True)
    combined_data.to_csv(f"{DATA_PATH}/data.csv", index=False)
    print(f"💾 data.csv saved to: {DATA_PATH}")


@time_it
def load_local_data():
    df = pd.read_csv(f"{DATA_PATH}/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df


if __name__ == "__main__":
    print(f"{Fore.WHITE}{Back.BLACK}[+] GP Patient Survey - MAKE DATA")
    monitor.ping(state="run")
    # Load new data from Google Sheet
    raw_data = load_google_sheet()

    # Load local data.csv to dataframe
    processed_data = load_local_data()

    # Return new data for processing
    data = raw_data[~raw_data.index.isin(processed_data.index)]

    print(f"{Fore.BLUE}[*] New rows to process: {data.shape[0]}")
    if data.shape[0] != 0:

        columns_to_sum = [
            "phone",
            "appointment_time",
            "making_appointment",
            "overall_experience",
            "website",
        ]
        data = add_row_values(data, columns_to_sum)

        data = word_count(data)  # word count
        data = textblob_sentiment(data)
        data["free_text"] = data["free_text"].apply(anonymize_names_with_transformers)
        data = feedback_classification(data, batch_size=16)

        concat_save_final_df(processed_data, data)
        do_git_merge()  # Push everything to GitHub
        monitor.ping(state="complete")
    else:
        print(f"{Fore.RED}[*] No New rows to add - terminated.")
        monitor.ping(state="complete")
