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


@st.cache_data(ttl=100)  # This decorator enables caching for this function
def get_surgery_data(data, selected_surgery):
    # Extracting unique surgery types
    surgery_list = data["surgery"].unique()

    # Filtering the dataset based on the selected surgery type
    surgery_data = data[data["surgery"] == selected_surgery]
    return surgery_data


surgery_list = data["surgery"].unique()
selected_surgery = st.sidebar.selectbox("Select Surgery", surgery_list)

# Call the function with the selected surgery
surgery_data = get_surgery_data(data, selected_surgery)

st.sidebar.container(height=5, border=0)

page = st.sidebar.radio(
    "Choose a Page",
    [
        "Dashboard",
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
if page == "Dashboard":
    st.header('GP Patient Survey Responses')
    cols = st.columns(2)
    with cols[0]:
        ui.metric_card(title="2024 PCN Survey - Total Responses", content=f"{surgery_data.shape[0]}", description=f"6.25% op practice list size.", key="total")
    with cols[1]:
        ui.metric_card(title="2023 National GP Pt Survey Responses", content=f"116", description=f"2.9% op practice list size.", key="total2")
        
        
    st.subheader("Q1. Generally, how easy is it to get through to someone at your GP practice on the phone?")
    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(y=surgery_data["phone"], ax=ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Phone")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(width + 0.3,  # Set the text at 0.3 unit right of the bar
                p.get_y() + p.get_height() / 2,  # Set the text at the center of the bar's height
                f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
                va = 'center')  # Center the text vertically
    st.pyplot(plt)

    st.subheader("Q2. How satisfied are you with the general practice appointment times that are available to you?")
    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(y=surgery_data["appointment_time"], ax=ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Appointment Times")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(width + 0.3,  # Set the text at 0.3 unit right of the bar
                p.get_y() + p.get_height() / 2,  # Set the text at the center of the bar's height
                f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
                va = 'center')  # Center the text vertically
    st.pyplot(plt)

    st.subheader("Q3. Overall, how would you describe your experience of making an appointment?")
    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(y=surgery_data["making_appointment"], ax=ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Making an Appointment")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(width + 0.3,  # Set the text at 0.3 unit right of the bar
                p.get_y() + p.get_height() / 2,  # Set the text at the center of the bar's height
                f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
                va = 'center')  # Center the text vertically
    st.pyplot(plt)

    st.subheader("Q4. Overall, how would you describe your experience of your GP practice?")
    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(y=surgery_data["overall_experience"], ax=ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Experience GP Practice")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(width + 0.3,  # Set the text at 0.3 unit right of the bar
                p.get_y() + p.get_height() / 2,  # Set the text at the center of the bar's height
                f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
                va = 'center')  # Center the text vertically
    st.pyplot(plt)

    st.subheader("Q5. Overall, how easy is your GP practice's website to use?")
    fig, ax = plt.subplots(figsize=(12, 4))  # Width=12, Height=4
    sns.countplot(y=surgery_data["website"], ax=ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.set_title("Surgery Website")
    ax.set_ylabel("")
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar
        ax.text(width + 0.3,  # Set the text at 0.3 unit right of the bar
                p.get_y() + p.get_height() / 2,  # Set the text at the center of the bar's height
                f"{int(width)} / {round((int(width)/surgery_data.shape[0])*100, 1)}%",
                va = 'center')  # Center the text vertically
    st.pyplot()

# == Rating & Sentiment Analysis Correlation ======================================================================
elif page == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )

    # React to the toggle's state
    if toggle:
        st.markdown(
            """1. **Scatter Plot (Top Plot)**:
This plot compares patient feedback sentiment scores with feedback rating scores. On the x-axis, we have the rating score, which likely corresponds to a numerical score given by the patient in their feedback, and on the y-axis, we have the sentiment score, which is derived from sentiment analysis of the textual feedback provided by the patient. Each point represents a piece of feedback, categorized as 'positive', 'neutral', or 'negative' sentiment, depicted by different markers. The scatter plot shows a clear positive correlation between the sentiment score and the feedback rating score, especially visible with the concentration of 'positive' sentiment scores at the higher end of the rating score scale, suggesting that more positive text feedback corresponds to higher numerical ratings.
2. **Histogram with a Density Curve (Bottom Left - NEGATIVE Sentiment)**:
This histogram displays the distribution of sentiment scores specifically for negative sentiment feedback. The x-axis represents the sentiment score (presumably on a scale from 0 to 1), and the y-axis represents the count of feedback instances within each score range. The bars show the frequency of feedback at different levels of negative sentiment, and the curve overlaid on the histogram provides a smooth estimate of the distribution. The distribution suggests that most negative feedback has a sentiment score around 0.7 to 0.8.
3. **Histogram with a Density Curve (Bottom Right - POSITIVE Sentiment)**:
Similar to the negative sentiment histogram, this one represents the distribution of sentiment scores for positive sentiment feedback. Here, we see a right-skewed distribution with a significant concentration of feedback in the higher sentiment score range, particularly close to 1.0. This indicates that the positive feedback is often associated with high sentiment scores, which is consistent with the expected outcome of sentiment analysis.
4. **View Patient Feedback (Multi-Select Input)**:
Select Patient feedback to review, this page only displays feedback that on Sentiment Analysis scored **NEGATIVE > Selected Value (using slider)**, indicating negative feedback despite rating given by the patient. It is very important to review feedback with a high NEG sentiment analysis. In this section both feedback and Improvement Suggestions are displayed to review them in context, together with the automated category assigned by our machine learning model."""
        )

    # Data for plotting
    labels = "Positive", "Neutral", "Negative"
    sizes = sentiment_totals(surgery_data)
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

    plt.title("Patient Feedback Sentiment Distribution")
    st.pyplot(fig)

    # Resample and count the entries per month from filtered data
    weekly_sent = surgery_data.resample("W", on="time")[
        "neg", "pos", "neu", "compound"
    ].mean()
    weekly_sent_df = weekly_sent.reset_index()
    weekly_sent_df.columns = ["Week", "neg", "pos", "neu", "compound"]
    weekly_sent_df["Week"] = pd.to_datetime(weekly_sent_df["Week"])

    @st.cache_data(ttl=100)  # This decorator caches the output of this function
    def calculate_weekly_sentiment(data):
        """
        Calculate the weekly sentiment averages from the given DataFrame.

        Parameters:
        data (DataFrame): The DataFrame containing sentiment scores and time data.

        Returns:
        DataFrame: A DataFrame with weekly averages of sentiment scores.
        """
        # Resample the data to a weekly frequency and calculate the mean of sentiment scores
        weekly_sent = data.resample("W", on="time")[
            "neg", "pos", "neu", "compound"
        ].mean()

        # Reset the index to turn the 'time' index into a column and rename columns
        weekly_sent_df = weekly_sent.reset_index()
        weekly_sent_df.columns = ["Week", "neg", "pos", "neu", "compound"]

        # Convert the 'Week' column to datetime format
        weekly_sent_df["Week"] = pd.to_datetime(weekly_sent_df["Week"])

        return weekly_sent_df


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
    st.subheader("About (FFT Dashboard)")
    # st.image(
    #     "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftestabout.png?raw=true",
    #     use_column_width=True,
    # )

    st.markdown(
        """Welcome to our new dashboard, aimed at enhancing how healthcare providers understand and use patient feedback. This tool focuses on the Friends and Family Test (FFT), which is essential for collecting patients' views on healthcare services. Our approach uses advanced text classification and sentiment analysis to organize and interpret this feedback more effectively.

Here's the core idea: Instead of just counting responses, the dashboard analyzes the sentiments behind them—whether positive, negative, or neutral. It assigns a detailed score to each piece of feedback, allowing for a more nuanced understanding of patient satisfaction. This method helps identify specific areas needing improvement and those that are performing well, based on real patient experiences.

For healthcare providers, this tool offers a more insightful way to look at patient feedback. It doesn’t just provide data; it offers a clearer picture of how patients feel about their care. This can help highlight excellence in services and pinpoint areas for potential improvements.

The data we use comes from a GP surgery in West London, showing how this tool can be applied in a real healthcare setting.

We employ several machine learning techniques for analysis:

1. **Sentiment Analysis:** Using Huggingface's 'cardiffnlp/twitter-roberta-base-sentiment-latest' model, we determine the emotional tone of the feedback.
2. **Text Classification** of Patient Feedback: To categorize feedback into different emotional themes, we use the 'SamLowe/roberta-base-go_emotions' model from Huggingface.
3. **Zero-shot Classification** of Patient Improvement Suggestions: The 'facebook/bart-large-mnli' model helps us identify and classify suggestions for improving patient care, even when the model hasn’t been specifically trained on healthcare data.
4. **Fine-tuned Zero-shot Calssification with FitSet** Classification of GP Reviews achieved with SetFit Algorithm. SetFit first fine-tunes a Sentence Transformer model on a small number of labeled examples (8 per class). This is followed by training a classifier head on the embeddings generated from the fine-tuned Sentence Transformer. https://huggingface.co/blog/setfit
5. Visit [**AI MedReview**](https://github.com/janduplessis883/friends-and-family-test-analysis) on GitHub, collaboration welcomed."""
    )

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
    st.subheader("GPT4 Summary")
    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    if toggle:
        st.markdown(
            """**What This Page Offers:**

**Automated Summaries**: Leveraging OpenAI's cutting-edge ChatGPT-4, we transform the Friends & Family Test feedback and improvement suggestions into concise, actionable insights.  
**Time-Specific Insights**: Select the period that matters to you. Whether it's a week, a month, or a custom range, our tool distills feedback relevant to your chosen timeframe.  
**Efficient Meeting Preparations**: Prepare for meetings with ease. Our summaries provide a clear overview of patient feedback, enabling you to log actions and decisions swiftly and accurately.  

**How It Works**:

1. **Select Your Time Period**: Choose the dates that you want to analyze.  
2. **AI-Powered Summarization**: ChatGPT-4 reads through the feedback and suggestions, understanding the nuances and key points.  
3. **Receive Your Summary**: Get a well-structured, comprehensive summary that highlights the core sentiments and suggestions from your patients."""
        )

    surgery_data = surgery_data[
        (surgery_data["time"].dt.date >= selected_date_range[0])
        & (surgery_data["time"].dt.date <= selected_date_range[1])
    ]
    surgery_data["prompt"] = surgery_data["free_text"].str.cat(
        surgery_data["do_better"], sep=" "
    )
    series = pd.Series(surgery_data["prompt"])
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
