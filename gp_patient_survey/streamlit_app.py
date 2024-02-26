import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from datetime import datetime
from datetime import date
from matplotlib.patches import Patch
import time
from openai import OpenAI
import streamlit_shadcn_ui as ui
import os

client = OpenAI()

from utils import *

st.set_page_config(page_title="AI MedReview: GP Patient Survey")


html = """
<style>
.gradient-text {
    background: linear-gradient(45deg, #163960, #598dac, #95c0d6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 2em;
    font-weight: bold;
}
</style>
<div class="gradient-text">GP Patient Survey</div>

"""
# Using the markdown function with HTML to center the text
st.sidebar.markdown(html, unsafe_allow_html=True)

st.sidebar.image(
    "https://github.com/janduplessis883/gp-patient-survey/blob/master/images/gps.png?raw=true"
)


@st.cache_data(ttl=100)
def load_data():
    df = pd.read_csv("gp_patient_survey/data/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df


data = load_data()

surgery_data = [
    {"surgery": "Earls Court Medical Centre", "list_size": 6817, "prev_survey": 103},
    {"surgery": "Earls Court Surgery", "list_size": 4129, "prev_survey": 113},
    {"surgery": "Emperor's Gate Health Centre", "list_size": 6909, "prev_survey": 103},
    {
        "surgery": "Health Partners at Violet Melchett",
        "list_size": 8000,
        "prev_survey": 103,
    },
    {"surgery": "Knightsbridge Medical Centre", "list_size": 17495, "prev_survey": 127},
    {"surgery": "Royal Hospital Chelsea", "list_size": 250, "prev_survey": 100},
    {"surgery": "Stanhope Mews Surgery", "list_size": 16277, "prev_survey": 194},
    {"surgery": "The Abingdon Medical Practice", "list_size": 8855, "prev_survey": 113},
    {"surgery": "The Chelsea Practice", "list_size": 6270, "prev_survey": 122},
    {"surgery": "The Good Practice", "list_size": 8029, "prev_survey": 105},
    {"surgery": "Scarsdale Medical Centre", "list_size": 8980, "prev_survey": 101},
]
# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(surgery_data)


@st.cache_data(ttl=100)  # This decorator enables caching for this function
def get_surgery_data(data, selected_surgery):
    # Extracting unique surgery types
    surgery_list = data["surgery"].unique()

    # Filtering the dataset based on the selected surgery type
    surgery_data = data[data["surgery"] == selected_surgery]
    return surgery_data


surgery_list = data["surgery"].unique()
surgery_list.sort()
selected_surgery = st.sidebar.selectbox("Select Surgery", surgery_list)

# Call the function with the selected surgery
surgery_data = get_surgery_data(data, selected_surgery)
# list_size = next((surgery["list_size"] for surgery in surgery_data if surgery["surgery"] == selected_surgery), None)
st.sidebar.container(height=5, border=0)

page = st.sidebar.radio(
    "Choose a Page",
    [
        "Survey Outcome",
        "Feedback Classification",
        "Sentiment Analysis",
        "GPT4 Summary",
        "View Dataframe",
        "About",
    ],
)
st.sidebar.container(height=200, border=0)


st.sidebar.write("")

centered_html = """
    <style>
    .centered {
        text-align: center;
    }
    </style>
    <div class='centered'>
    <img alt="Static Badge" src="https://img.shields.io/badge/github-janduplessis883-%23d0ae57?logo=github&color=%23d0ae57&link=https%3A%2F%2Fgithub.com%2Fjanduplessis883%2Ffriends-and-family-test-analysis">
    </div>
"""


# Using the markdown function with HTML to center the text
st.sidebar.markdown(centered_html, unsafe_allow_html=True)


# == DASHBOARD ==========================================================================================================
if page == "Survey Outcome":
    st.header(f"{selected_surgery}")

    list_size = df.loc[df["surgery"] == selected_surgery, "list_size"].values[0]
    prev_survey = df.loc[df["surgery"] == selected_surgery, "prev_survey"].values[0]
    st.markdown(f"List size: `{list_size}`")

    cols = st.columns(2)
    with cols[0]:
        ui.metric_card(
            title="2024 PCN Patient Survey - Responses",
            content=f"{surgery_data.shape[0]}",
            description=f"{round(surgery_data.shape[0]/list_size*100, 2)}% of practice list size.",
            key="total",
        )
    with cols[1]:
        ui.metric_card(
            title="2023 NHS GP Patient Survey - Responses",
            content=f"{prev_survey}",
            description=f"{round(prev_survey/list_size*100, 2)}% of practice list size.",
            key="total2",
        )
    st.markdown("---")
    order1 = [
        "Very easy",
        "Fairly easy",
        "Not very easy",
        "Not at all easy",
    ]

    palette1 = {
        "Very easy": "#204e82",
        "Fairly easy": "#204e82",
        "Not very easy": "#95c0d6",
        "Not at all easy": "#95c0d6",
    }

    st.subheader(
        "Q1. Generally, how easy is it to get through to someone at your GP practice on the phone?"
    )
    counts = surgery_data["phone"].value_counts()
    to_count1 = counts["Very easy"]
    to_count2 = counts["Fairly easy"]
    st.markdown(
        f":orange[**{round((to_count1 + to_count2)/surgery_data.shape[0]*100, 2)}%** find it easy to get through to this GP practice by phone] "
    )
    st.markdown(":grey[ICS result: 59% National result: 50%]")

    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(y=surgery_data["phone"], ax=ax, order=order1, palette=palette1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Phone")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(
            width + 0.3,  # Set the text at 0.3 unit right of the bar
            p.get_y()
            + p.get_height() / 2,  # Set the text at the center of the bar's height
            f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
            va="center",
        )  # Center the text vertically
    st.pyplot(plt)
    st.markdown("---")
    st.subheader(
        "Q2. How satisfied are you with the general practice appointment times that are available to you?"
    )

    order3 = [
        "Very satisfied",
        "Fairly satisfied",
        "Neither satisfied nor dissatisfied",
        "Fairly dissatisfied",
        "Very dissatisfied",
    ]

    palette3 = {
        "Very satisfied": "#204e82",
        "Fairly satisfied": "#204e82",
        "Neither satisfied nor dissatisfied": "#95c0d6",
        "Fairly dissatisfied": "#95c0d6",
        "Very dissatisfied": "#95c0d6",
    }

    counts = surgery_data["appointment_time"].value_counts()
    to_count1 = counts["Very satisfied"]
    to_count2 = counts["Fairly satisfied"]
    st.markdown(
        f":orange[**{round((to_count1 + to_count2)/surgery_data.shape[0]*100, 2)}%** are satisfied with the general practice appointment times available.] "
    )
    st.markdown(":grey[ICS result: 57% National result: 53%]")
    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(
        y=surgery_data["appointment_time"], ax=ax, order=order3, palette=palette3
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Appointment Times")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(
            width + 0.3,  # Set the text at 0.3 unit right of the bar
            p.get_y()
            + p.get_height() / 2,  # Set the text at the center of the bar's height
            f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
            va="center",
        )  # Center the text vertically
    st.pyplot(plt)
    st.markdown("---")
    order = [
        "Very good",
        "Fairly good",
        "Neither good nor poor",
        "Fairly poor",
        "Very poor",
    ]

    palette = {
        "Very good": "#204e82",
        "Fairly good": "#204e82",
        "Neither good nor poor": "#95c0d6",
        "Fairly poor": "#95c0d6",
        "Very poor": "#95c0d6",
    }

    st.subheader(
        "Q3. Overall, how would you describe your experience of making an appointment?"
    )

    counts = surgery_data["making_appointment"].value_counts()
    to_count1 = counts["Very good"]
    to_count2 = counts["Fairly good"]
    st.markdown(
        f":orange[**{round((to_count1 + to_count2)/surgery_data.shape[0]*100, 2)}%** describe their experience of making an appointment as good.] "
    )
    st.markdown(":grey[ICS result: 57% National result: 54%]")
    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(
        y=surgery_data["making_appointment"], ax=ax, order=order, palette=palette
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Making an Appointment")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(
            width + 0.3,  # Set the text at 0.3 unit right of the bar
            p.get_y()
            + p.get_height() / 2,  # Set the text at the center of the bar's height
            f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
            va="center",
        )  # Center the text vertically
    st.pyplot(plt)
    st.markdown("---")
    st.subheader(
        "Q4. Overall, how would you describe your experience of your GP practice?"
    )

    counts = surgery_data["overall_experience"].value_counts()
    to_count1 = counts["Very good"]
    to_count2 = counts["Fairly good"]
    st.markdown(
        f":orange[**{round((to_count1 + to_count2)/surgery_data.shape[0]*100, 2)}%** describe their overall experience of this GP practice as good.]"
    )
    st.markdown(":grey[ICS result: 70% National result: 71%]")
    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(
        y=surgery_data["overall_experience"], ax=ax, order=order, palette=palette
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Experience GP Practice")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(
            width + 0.3,  # Set the text at 0.3 unit right of the bar
            p.get_y()
            + p.get_height() / 2,  # Set the text at the center of the bar's height
            f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
            va="center",
        )  # Center the text vertically
    st.pyplot(plt)
    st.markdown("---")
    st.subheader("Q5. Overall, had a good experiece using the surgery website.")

    counts = surgery_data["website"].value_counts()
    to_count1 = counts["Very good"]
    to_count2 = counts["Fairly good"]
    st.markdown(
        f":orange[**{round((to_count1 + to_count2)/surgery_data.shape[0]*100, 2)}%** had a good experiece using the surgery website.]"
    )
    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(y=surgery_data["website"], ax=ax, order=order, palette=palette)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Surgery Website")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(
            width + 0.3,  # Set the text at 0.3 unit right of the bar
            p.get_y()
            + p.get_height() / 2,  # Set the text at the center of the bar's height
            f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
            va="center",
        )  # Center the text vertically
    st.pyplot()

# == Rating & Sentiment Analysis Correlation ======================================================================
elif page == "Sentiment Analysis":
    st.subheader("Free-Text Feedback Sentiment Analysis")
    st.container(height=15, border=False)
    filtered_data = surgery_data[pd.notna(surgery_data["free_text"])]

    # Data for plotting
    labels = "Positive", "Neutral", "Negative"
    sizes = sentiment_totals(filtered_data)
    colors = ["#6b899f", "#f0e8d2", "#ae4f4d"]
    explode = (0, 0, 0)  # 'explode' the 1st slice (Positive)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    fig.gca().add_artist(centre_circle)

    plt.title("Cumulative Sentiment Analysis Overview")
    st.pyplot(fig)


# == Feedback Classification ========================================================================================
elif page == "Feedback Classification":
    st.subheader("Feedback Classification")

    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    if toggle:
        st.markdown(
            """1. **Bar Chart**:
This bar chart illustrates the range of emotions captured in the FFT feedback, as categorized by a sentiment analysis model trained on the `go_emotions` dataset. Each bar represents one of the 27 emotion labels that the model can assign, showing how often each emotion was detected in the patient feedback.
The **'neutral' category**, which has been assigned the most counts, includes instances where patients did not provide any textual feedback, defaulting to a 'neutral' classification. Other emotions, such as 'admiration' and 'approval', show varying lower counts, reflecting the variety of sentiments expressed by patients regarding their care experiences.

2. **Multi-select Input Field**:
Below the chart is a multi-select field where you can choose to filter and review the feedback based on these emotion labels. This feature allows you to delve deeper into the qualitative data, understanding the nuances behind the ratings patients have given and potentially uncovering areas for improvement in patient experience."""
        )

    # Calculate value counts
    label_counts = surgery_data["feedback_labels"].value_counts(
        ascending=False
    )  # Use ascending=True to match the order in your image

    # Convert the Series to a DataFrame
    label_counts_df = label_counts.reset_index()
    label_counts_df.columns = ["Feedback Classification", "Counts"]

    # Define the palette conditionally based on the category names
    palette = [
        "#aec867" if (label == "Overall Patient Satisfaction") else "#62899f"
        for label in label_counts_df["Feedback Classification"]
    ]

    # Create a Seaborn bar plot
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        x="Counts", y="Feedback Classification", data=label_counts_df, palette=palette
    )
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    # Adding titles and labels for clarity
    plt.title("Counts of Feedback Classification")
    plt.xlabel("Counts")
    plt.ylabel("")

    # Streamlit function to display matplotlib figures
    st.pyplot(plt)

    # View Patient Feedback
    st.subheader("View Patient Feedback")
    class_list = list(surgery_data["feedback_labels"].unique())
    cleaned_class_list = [x for x in class_list if not pd.isna(x)]
    selected_ratings = st.multiselect("Select Feedback Categories:", cleaned_class_list)

    # Filter the data based on the selected classifications
    filtered_classes = surgery_data[
        surgery_data["feedback_labels"].isin(selected_ratings)
    ]

    if not selected_ratings:
        ui.badges(
            badge_list=[("Please select at least one classification.", "outline")],
            class_name="flex gap-2",
            key="badges10",
        )
    else:
        for rating in selected_ratings:
            specific_class = filtered_classes[
                filtered_classes["feedback_labels"] == rating
            ]
            st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
            for text in specific_class[
                "free_text"
            ]:  # Assuming 'free_text' is the column with the text you want to display
                if str(text).lower() != "nan" and str(text).lower() != "neutral":
                    st.write("- " + str(text))


# == Dataframe ==========================================================
elif page == "View Dataframe":
    st.subheader("Dataframe")
    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    if toggle:
        st.markdown(
            """**Dataframe**:
A dataFrame as a big, organized table full of raw data. It's like a virtual spreadsheet with many rows and columns, where every row represents a single record, and each column stands for a particular variable. If your DataFrame contains all the raw data, it means that it hasn't been processed or filtered - it's the data in its original form as collected.

Each column in a DataFrame has a name, which you can use to locate data more easily. Columns can contain all sorts of data types, including numbers, strings, and dates, and each one typically holds the same kind of data throughout. For instance, one column might hold ages while another lists names, and yet another records dates of visits.

Rows are labeled with an Index, which you can think of as the address of the data. This makes finding specific records simple and fast."""
        )
    st.write("The data below is filtered based on the date range selected above.")

    # Display the filtered DataFrame
    st.dataframe(surgery_data)

# == About ==========================================================
elif page == "About":
    st.subheader("About")
    # st.image(
    #     "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftestabout.png?raw=true",
    #     use_column_width=True,
    # )

    st.markdown("""Brompton Health PCN - GP Patient Survey""")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(y="surgery", data=data, color="#536570")
    for p in ax.patches:
        width = p.get_width()
        try:
            y = p.get_y() + p.get_height() / 2
            ax.text(
                width + 1,
                y,
                f"{int(width)}",
                va="center",
                fontsize=8,
            )
        except ValueError:
            pass
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    st.pyplot(plt)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # Use 'col1' to display content in the first column
    with col1:
        st.image(
            "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/about.png?raw=true",
            width=200,
        )

    # Use 'col2' to display content in the second column
    with col2:
        st.image(
            "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/hf-logo-with-title.png?raw=true",
            width=200,
        )
    with col3:
        st.image(
            "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/openai.png?raw=true",
            width=200,
        )


# == Generate ChatGPT Summaries ==========================================================
elif page == "GPT4 Summary":
    st.subheader("GPT4 Feedback Summary")

    series = pd.Series(surgery_data["free_text"])
    series.dropna(inplace=True)
    word_series = series.to_list()
    text = " ".join(word_series)
    text = replace_surname(text)

    def call_chatgpt_api(text):
        # Example OpenAI Python library request
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. and expert at summarixing friends and family Test Feedback for a GP Surgery",
                },
                {"role": "user", "content": f"Summarize the follwing text\n\n{text}"},
            ],
        )

        output = completion.choices[0].message.content
        return output

    # Text input
    user_input = text

    # Button to trigger summarization
    if st.button("Summarize with GPT4", help="Click to start generating a summary."):
        if user_input:
            # Call the function to interact with ChatGPT API
            st.markdown("### Input Text")
            code = text
            st.info(f"{code}")

            # Initiate progress bar
            my_bar = st.progress(0)

            # Simulate a loading process
            for percent_complete in range(100):
                time.sleep(0.2)
                my_bar.progress(percent_complete + 1)

            summary = call_chatgpt_api(user_input)

            # Hide the progress bar after completion
            my_bar.empty()
            st.markdown("---")
            st.markdown("### GPT4 Feedback Summary")
            st.markdown("`Copy GPPT4 Summary as required.`")
            st.write(summary)
            st.download_button(
                "Download GPT-4 Output", summary, help="Download summary as a TXT file."
            )

        else:
            st.write(text)
            ui.badges(
                badge_list=[("Not able to summarise text.", "destructive")],
                class_name="flex gap-2",
                key="badges10",
            )
    else:
        st.image(
            "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/unnamed.jpg?raw=true"
        )
