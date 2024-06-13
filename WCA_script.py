
import numpy as np
import pandas as pd
import re
import os
from datetime import datetime
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import MaxNLocator
# from nltk.corpus import stopwords
from collections import Counter , defaultdict, deque
import networkx as nx
from networkx.utils import groups
from networkx.algorithms import community
from wordcloud import WordCloud
import timeit
import graphs
from emoji import demojize
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

"""All Import Packages"""


"""Parse Text from Folder"""
"""Updated Text File Parsing from above code - use this now onwards
### Convert text file lines into each single line with
### Date format as the beginning of each line marked with \n in the end
"""
def format_txt_file(input_folder_path, output_folder_path):
    txt_file_name = "parsed_text_file_raw.txt"

    # List all files in the specified folder
    all_files = os.listdir(input_folder_path)

    # Filter out only the text files
    text_files = [file for file in all_files if file.endswith('.txt')]

    modified_lines = []

    # Loop through each text file
    for file_name in text_files:

        if file_name == str(txt_file_name):
            continue
        #print(f"reading {file_name}")

        file_path = os.path.join(input_folder_path, file_name)

        # Open the text file and read its contents
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Process each line in the text file
        for line in lines:
            line = line.strip()
            # if re.match(r'^\b\d{1,2}\/\d\d\/\d\d, \b', line):
            if re.match(r'^\d{1,2}/\d{1,2}/\d{2},', line):
                # #print("re matched: ", line)
                # if modified_lines:
                #     modified_lines.append('\n')  # Add new line before date-time line
                modified_lines.append(line)
            elif modified_lines:  # Check if there are previous lines to concatenate
                modified_lines[-1] += ' ' + line
            else:
                modified_lines.append(line)

    # Write the modified lines to the output file
    with open(os.path.join(output_folder_path, txt_file_name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(modified_lines))


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""## Data Extraction and Preprocessing
### Date and Time Extraction from Txt Files"""

"""Bool Function for Date Time Extraction"""


def dateTime(line):
    reg_pattern = r'\b\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{1,2}(?:\s?[AP]M)?\b'

    date_time = re.findall(reg_pattern, line)
    modified_line = re.sub(reg_pattern, '', line).strip()
    if date_time:
        # #print(date_time)
        date_time_str = date_time[0]
        date_str, time_str = re.split(r',\s*', date_time_str)
        time_str = re.sub(r'\s*[\u202f]+', '', time_str)

        if modified_line:
            return [date_str, time_str, modified_line]
        if date_str and time_str and not modified_line:
            return [date_str, time_str, None]

    else:
        return [None, None, None,]


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""### Name and Message Extraction"""
"""Bool Function for Author Name Extraction"""


def AuthorName(line):
    # #print('\nline = ',line)
    # reg_pattern = r'([\w\s]+|\+\d{2,}\s\d{5}\s\d{5}):'
    reg_pattern = r'([\w\s\+,]+):\s'  # r'([\w\s\+]+):\s'
    matches = re.findall(reg_pattern, line)
    if matches:
        author_name = matches[0]  # Assuming only one author name per line
        modified_line = re.sub(reg_pattern, '', line, count=1).strip()
        # #print("Author Name:", author_name)
        # #print("Modified Line:", modified_line)
        return author_name, modified_line
    else:
        return None, line.strip()


def AuthorName2(line):
    parts = line.split(': ', 1)
    if len(parts) == 2:
        author_name2 = parts[0]
        modified_line2 = parts[1]
        return author_name2, modified_line2
    else:
        return None, line.strip()


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""


"""### PARSE and extract the data and Format it into Date, Time, Name, Message
"""
"""Import Whatsapp Group chats and store them
data sample: 3/24/21, 2:32 PM - UserName: text, emojis 🤓, <media ommitted>
4 attributes — Date, Time, Author, Message.
"""
# ParsedData = []
# errorline = []


def parse_raw_txt_file(path, ParsedData=[], errorline=[]):

    count_error = 0
    passed = 0

    with open(path, 'r', encoding='utf-8') as fp:

        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            # #print(line)
            if (len(line) <= 0):
                continue

            try:
                
                dtl = list()
                dtl = dateTime(line)

                if (None in dtl):
                    raise TypeError

                passed += 1
                dtl[2] = dtl[2].replace('- ', '')

                author, msg = AuthorName(dtl[2])
                if (author == None):
                    author, msg = AuthorName2(dtl[2])

                ParsedData.append([dtl[0], dtl[1], author, msg])

            except:
                count_error += 1
                errorline.append(line)

    if (count_error > 0):
        print(f'\nCoudn\'t Parse {count_error} lines due to TypeError')
    else:
        # 126834
        print(f"Successfully Parsed {passed} lines")
    #print("\nParsed Data:\n")
    return ParsedData


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""## Convert the Parsed Data list into Pandas DataFrame
### For Further Analysis - Created Original DataFrame : DF"""


def initial_preprocessing(ParsedData):

    df = pd.DataFrame(ParsedData, columns=['Date', 'Time', 'Name', 'Message'])
    """Dropping the Null Value Names"""
    # null_name_df = df[df['Name'].isnull()]
    df = df.dropna(subset=['Name'])
    """df after null name drops"""
    # df.describe()
    # df[df['Name'].isnull()]
    return df


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""Formatting the date as datetime object"""
# Custom date parsing function


def parse_date(date_str):
    # List of possible date formats
    date_formats = [
        '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',  # Day first formats
        '%m/%d/%y', '%m-%d-%y', '%m.%d.%y',  # Month first formats
        '%y/%m/%d', '%y-%m-%d', '%y.%m.%d',  # Year first formats
    ]

    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime('%d/%m/%Y')
        except ValueError:
            continue
    # If no format matches, return None or raise an error
    return None


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""Formatting the Time as datetime object"""

# Custom time parsing function


def parse_time(time_str):
    # List of possible time formats
    time_formats = [
        '%I:%M %p',  # 12-hour format with AM/PM (e.g., 9:52 AM)
        '%I:%M%p',   # 12-hour format without space before AM/PM (e.g., 9:52AM)
        '%H:%M',     # 24-hour format (e.g., 10:52)
    ]

    for fmt in time_formats:
        try:
            return datetime.strptime(time_str, fmt).strftime('%H:%M')
        except ValueError:
            continue
    # If no format matches, return None or raise an error
    return None


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

# return df


def Message_Frequency_Analysis(df):
    """Count the total number of messages in the DF."""
    total_msg_count = df['Message'].count()
    # #print(f"the total number of messages in the DF: {total_msg_count}")

    """Calculate the total and average number of messages per day.

        This gives you the total number of messages for each day, 
        which includes all the messages sent at any hour on that day.
    """
    # Grouping By day and then getting the msg count and rounding off
    total_msg_per_day = df.groupby(df['Date'].dt.date)[
        'Message'].count()  # List - day : , msg_count
    avg_msg_per_day = np.round(total_msg_per_day.mean())
    # #print(f"\nthe total number of messages per Day:\n {total_msg_per_day[:3]}")
    # #print(f'\n\nAverage (Mean) Messages sent in a day (Approx) : {avg_msg_per_day}')
    return total_msg_per_day


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""
"""Error
Locator attempting to generate 1898
 ticks ([18187.0, ..., 20084.0]),
 which exceeds Locator.MAXTICKS (1000).
"""


def IQR(total_msg_per_day):
    """Line Chart plot"""
    """sO IT SEEMS FOR SMALL MESSAGE COUNTS iqr IS NOT FEASIBLE - GOING TO NEGETIVE AND LEN = 0 """
    # Copy the variable
    daily_msg_count = total_msg_per_day
    # #print("len of msg before: ", len(total_msg_per_day))

    # Number of entries to consider for the last 20%
    num_entries_20_percent = int(len(daily_msg_count) * 0.20)
    # #print("num_entries_20_percent: ", num_entries_20_percent)

    # Get the last 20% of data
    last_20_percent_data = daily_msg_count[-num_entries_20_percent:]
    # #print("len of last_20_percent_data: ", len(last_20_percent_data))

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = last_20_percent_data.quantile(0.25)
    Q3 = last_20_percent_data.quantile(0.75)
    IQR = Q3 - Q1
    # #print("Q1: ", Q1, " Q3: ", Q3, " IQR: ", IQR)

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # #print("lower_bound: ", lower_bound, " upper_bound: ", upper_bound)

    # Filter out the outliers
    filtered_last_20_percent_data = last_20_percent_data[
        (last_20_percent_data >= lower_bound) & (last_20_percent_data <= upper_bound)]
    # #print("len of filtered_last_20_percent_data: ", len(filtered_last_20_percent_data))

    # Combine the filtered last 20% data with the rest of the data
    daily_msg_count_filtered = pd.concat(
        [daily_msg_count[:-num_entries_20_percent], filtered_last_20_percent_data])
    # Continue removing from the end until you encounter a message count of 100 or more
    while not daily_msg_count_filtered.empty and daily_msg_count_filtered.iloc[-1] < 110:
        daily_msg_count_filtered = daily_msg_count_filtered.iloc[:-1]
    # #print("len of msg after filtering: ", len(daily_msg_count_filtered))

    if (len(daily_msg_count_filtered) <= 50):
        daily_msg_count_filtered = total_msg_per_day
    return daily_msg_count_filtered


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""


def hourly_FA(df):
    """Get total hourly and monthly msg count"""
    hourly_msg_count = df.groupby(df['hour']).size(
    )  # Can be use to check peak messaging hour #List - hour : , msg_count
    """It doesn't consider the month or year, only the day within each month.
        all messages sent on a particular day of any month, regardless of the year, will be grouped together."""
    # daily_msg_count = df.groupby(df['day']).size() # hence Redundant
    # can be used to evaluate messaging based on each month's activity #List - month : , msg_count
    monthly_msg_count = df.groupby(df['month']).size()

    # #print(f"\nTotal Hourly messages Sent in whole DF: \n\t{hourly_msg_count.values}")
    # #print(f"\nTotal daily messages Sent: {daily_msg_count[:2]}")
    # #print(f"\nTotal monthly messages Sent in whole DF: \n\t{monthly_msg_count.values}")

    # #print(f"\nMost active hour of Messaging: {hourly_msg_count.idxmax()}:00 Hrs = {hourly_msg_count.max()}")
    # #print(f"\nMost active month of Messaging: month - {monthly_msg_count.idxmax()} = {monthly_msg_count.max()}")

    return hourly_msg_count, monthly_msg_count


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""


def hour_month_stats(hourly_msg_count, monthly_msg_count):
    """hourly Message statistical Analysis using hourly_msg_count above"""
    hourly_msg_count_mean = hourly_msg_count.mean()
    hourly_msg_count_median = hourly_msg_count.median()
    hourly_msg_count_mode = hourly_msg_count.mode()
    hourly_msg_count_std = hourly_msg_count.std()
    hourly_msg_count_variance = hourly_msg_count.var()

    # #print(f"hourly_msg_count_mean = {hourly_msg_count_mean}")
    # #print(f"hourly_msg_count_median = {hourly_msg_count_median}")
    # #print(f"hourly_msg_count_mode = \n{hourly_msg_count_mode.head()}")
    # #print(f"hourly_msg_count_std = {hourly_msg_count_std}")
    # #print(f"hourly_msg_count_variance = {hourly_msg_count_variance}")

    """Monthly Message statistical Analysis using monthly_msg_count above"""
    monthly_msg_count_mean = monthly_msg_count.mean()
    monthly_msg_count_median = monthly_msg_count.median()
    monthly_msg_count_mode = monthly_msg_count.mode()
    monthly_msg_count_std = monthly_msg_count.std()
    monthly_msg_count_variance = monthly_msg_count.var()

    # #print(f"monthly_msg_count_mean = {monthly_msg_count_mean}")
    # #print(f"monthly_msg_count_median = {monthly_msg_count_median}")
    # #print(f"monthly_msg_count_mode = \n{monthly_msg_count_mode.head()}")
    # #print(f"monthly_msg_count_std = {monthly_msg_count_std}")
    # #print(f"monthly_msg_count_variance = {monthly_msg_count_variance}")


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""


def week_analysis(df):

    # Step 1: Create week_df as before
    temp_week_dft = df.copy()
    temp_week_dft['DateTime'] = pd.to_datetime(temp_week_dft['DateTime'])

    # Group days into weeks
    temp_week_dft['week'] = (
        (temp_week_dft['DateTime'] - temp_week_dft['DateTime'].min()).dt.days // 7) + 1

    # Get peak messaging day per week
    msg_count_per_dayname = temp_week_dft.groupby(
        ['week', 'day name']).size().reset_index(name='message_count')
    peak_day_week = msg_count_per_dayname.loc[msg_count_per_dayname.groupby('week')[
        'message_count'].idxmax()]

    # Merge to get most_messaged_day and most_messaged_day_count
    unique_week_entries = temp_week_dft[['week', 'Date']].drop_duplicates(subset=[
                                                                          'week'])
    unique_week_entries = unique_week_entries.merge(peak_day_week[['week', 'day name', 'message_count']],
                                                    on='week', how='left')
    unique_week_entries.rename(columns={
                               'day name': 'most_messaged_day', 'message_count': 'most_messaged_day_count'}, inplace=True)

    # Handle missing values
    unique_week_entries.dropna(subset=['most_messaged_day'], inplace=True)
    unique_week_entries['most_messaged_day_count'] = unique_week_entries['most_messaged_day_count'].astype(
        int)

    # Step 2: Initialize lists to store grouped data
    week_ranges = []
    date_ranges = []
    most_messaged_days = []
    most_messaged_day_counts = []

    # Step 3: Update df with most_messaged_day and most_messaged_day_count
    df['week'] = temp_week_dft['week']
    df['most_messaged_day'] = temp_week_dft['day name']
    df['most_messaged_day_count'] = 0  # Initialize with 0 before updating

    # Step 4: Ensure message_count is included before calculating mean counts
    # Recompute temp_week_dft with message_count directly
    msg_count_per_dayname = temp_week_dft.groupby(
        ['week', 'day name']).size().reset_index(name='message_count')

    # Step 5: Update df with correct message_count values
    for week_num, group_df in msg_count_per_dayname.groupby('week'):
        for day_name, day_group in group_df.groupby('day name'):
            mean_count = day_group['message_count'].mean()
            df.loc[(df['week'] == week_num) & (df['day name'] ==
                                               day_name), 'most_messaged_day_count'] = mean_count

    # Step 6: Create week_df with the final grouped data
    current_start_week = unique_week_entries.iloc[0]['week']
    current_start_date = unique_week_entries.iloc[0]['Date']
    current_end_week = current_start_week
    current_end_date = current_start_date
    current_day = unique_week_entries.iloc[0]['most_messaged_day']
    current_counts = [unique_week_entries.iloc[0]
                      ['most_messaged_day_count']]  # Store counts in a list

    for i in range(1, len(unique_week_entries)):
        row = unique_week_entries.iloc[i]

        if row['most_messaged_day'] == current_day:
            # If the day matches, update the current end week and date and add count to list
            current_end_week = row['week']
            current_end_date = row['Date']
            current_counts.append(row['most_messaged_day_count'])
        else:
            # If they don't match, compute the mean of counts and store the current group
            mean_count = sum(current_counts) / len(current_counts)

            # Append the last group
            if current_start_week != current_end_week:
                week_ranges.append(f"{current_start_week}-{current_end_week}")
                date_ranges.append(
                    f"{current_start_date.date()}-{current_end_date.date()}")
            else:
                week_ranges.append(f"{current_start_week}")
                date_ranges.append(f"{current_start_date.date()}")

            most_messaged_days.append(current_day)
            most_messaged_day_counts.append(round(mean_count))

            # Start a new group
            current_start_week = row['week']
            current_start_date = row['Date']
            current_day = row['most_messaged_day']
            # Reset counts list with the new count
            current_counts = [row['most_messaged_day_count']]
            current_end_week = row['week']
            current_end_date = row['Date']

    # Process the last group
    mean_count = sum(current_counts) / len(current_counts)
    if current_start_week != current_end_week:
        week_ranges.append(f"{current_start_week}-{current_end_week}")
        date_ranges.append(
            f"{current_start_date.date()}-{current_end_date.date()}")
    else:
        week_ranges.append(f"{current_start_week}")
        date_ranges.append(f"{current_start_date.date()}")

    most_messaged_days.append(current_day)
    most_messaged_day_counts.append(round(mean_count))

    # Step 7: Create week_df with the final grouped data
    week_df = pd.DataFrame({
        'week_range': week_ranges,
        'date_range': date_ranges,
        'most_messaged_day': most_messaged_days,
        'most_messaged_day_count': most_messaged_day_counts
    })

    # Round off and convert to integer
    week_df['most_messaged_day_count'] = week_df['most_messaged_day_count'].round(
    ).astype(int)

    # Sort the DataFrame by ascending order of the week_range
    week_df.sort_values(by='week_range', inplace=True)
    week_df.reset_index(drop=True, inplace=True)

    # Calculate the 75th percentile
    threshold_75th_percentile = int(
        week_df['most_messaged_day_count'].quantile(0.75))

    # Filter the DataFrame to drop rows with counts less than the 75th percentile
    week_df_filtered = week_df[week_df['most_messaged_day_count']
                               >= threshold_75th_percentile]

    return week_df_filtered


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""Segment Users Based on Activity Level"""


def user_segmentation(df, most_active_users_list):

    # Define Threshold quantile: 0.25, 0.50, 0.90
    top_10_percentile = most_active_users_list.quantile(0.9)
    moderate_50_percentile = most_active_users_list.quantile(0.5)

    """categorize users based on high, moderate, low active level:"""

    # .index.tolist()   # need Scalar list? .index.tolist()
    high_active_level_users = most_active_users_list[most_active_users_list >=
                                                     top_10_percentile]
    Moderate_active_level_users = most_active_users_list[
        (most_active_users_list >= moderate_50_percentile) & (most_active_users_list < top_10_percentile)]  # .index.tolist()
    # .index.tolist()
    low_active_level_users = most_active_users_list[most_active_users_list <
                                                    moderate_50_percentile]

    # display top 2 for each category:
    # #print(f'High 2 : {high_active_level_users.head(2)}')
    # #print(f'\nMid 2 : {Moderate_active_level_users.head(2)}')
    # #print(f'\nLow 2 : {low_active_level_users.head(2)}')

    """ Now group them inorder oc activity levels of weekday and weekends """
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    weekend_msg_count = df[df['day name'].isin(weekends)].groupby('Name')[
        'Message'].count()
    weekday_msg_count = df[df['day name'].isin(weekdays)].groupby('Name')[
        'Message'].count()

    # Filter weekday and weekend message counts based on activity levels
    weekday_high = weekday_msg_count[weekday_msg_count.index.isin(
        high_active_level_users.index)]
    weekday_moderate = weekday_msg_count[weekday_msg_count.index.isin(
        Moderate_active_level_users.index)]
    weekday_low = weekday_msg_count[weekday_msg_count.index.isin(
        low_active_level_users.index)]

    weekend_high = weekend_msg_count[weekend_msg_count.index.isin(
        high_active_level_users.index)]
    weekend_moderate = weekend_msg_count[weekend_msg_count.index.isin(
        Moderate_active_level_users.index)]
    weekend_low = weekend_msg_count[weekend_msg_count.index.isin(
        low_active_level_users.index)]

    return [weekday_high, weekday_moderate, weekday_low, weekend_high, weekend_moderate, weekend_low]


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""Gather all the mentions"""

def extract_mentions(dfc):
    """code below"""
    # Create a copy for testing
    test_dfc2 = dfc.copy()

    # Regular expression pattern to extract phone numbers and mentions
    phone_pattern = re.compile(
        r'\b(?:\+?\d{1,3}[\s-]?)?(?:\d{1,4}[\s-]?){1,4}\d{1,4}\b')
    mention_pattern = re.compile(r'@(\w+)')
    email_pattern = re.compile(r'@([\w\.-]+)')

    name_words = set(test_dfc2['condensedName'].str.lower())

    def preprocess_message(message, min_len):
        # Remove words shorter than the minimum length
        return ' '.join([word for word in message.split() if len(word) >= min_len])

    def find_mentions(message):
        message_words = set(word.lower()
                            for word in re.findall(r'\b\w+\b', message))
        common_names = list(message_words.intersection(name_words))
        return common_names

    # Function to clean and validate phone numbers
    def clean_and_validate_phone_numbers(numbers):
        valid_numbers = []
        for num in numbers:
            clean_num = re.sub(r'\D', '', num)  # Remove non-digit characters
            if 10 <= len(clean_num) <= 15:
                valid_numbers.append(clean_num)
        return valid_numbers

    min_len = test_dfc2['condensedName'].str.len().min()
    #print(f"Minimum length of condensed names: {min_len}")

    for index, row in tqdm(test_dfc2.iterrows(), total=len(test_dfc2), desc="Processing chunk"):
        sender = row['condensedName']
        message = row['Message']

        # Preprocess message to remove short words
        processed_message = preprocess_message(message, min_len)

        mentions = set()

        # Extract mentions with @ symbol
        at_mentions = mention_pattern.findall(processed_message)
        mentions.update(at_mentions)

        # Extract email mentions
        email_mentions = email_pattern.findall(processed_message)
        mentions.update(email_mentions)

        # Extract potential phone numbers
        numbers = phone_pattern.findall(processed_message)

        # Clean and validate numbers and add to mentions
        valid_numbers = clean_and_validate_phone_numbers(numbers)
        mentions.update(valid_numbers)

        # Find mentions using set intersection
        mentions.update(find_mentions(processed_message))

        if mentions:
            test_dfc2.at[index, 'Mention'] = ', '.join(mentions)

    test_dfc2.dropna(subset=['Mention'], inplace=True)
    duplicates = test_dfc2[test_dfc2.duplicated(
        subset=['Name', 'Message', 'Mention'], keep=False)]
    test_dfc2.drop_duplicates(duplicates, inplace=True)

    return test_dfc2


"""Final Mention extract and delete function"""


def final_pure_mentions(filtered_ad):
    """
    Process mentions to ensure they are in list format, filter based on unique names,
    remove rows with empty mentions, and convert mentions back to string.

    Args:
    filtered_ad (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """

    def ensure_list_format(value):
        if isinstance(value, str):
            return value.split(',')
        elif isinstance(value, list):
            return value
        else:
            return []

    # Check and #print column names
    # #print("Columns in DataFrame:", filtered_ad.columns)

    # Ensure the 'Mention' column is in list format
    if 'Mention' in filtered_ad.columns:
        filtered_ad['Mention'] = filtered_ad['Mention'].apply(
            ensure_list_format)
    else:
        raise KeyError("'Mention' column not found in the DataFrame")

    # Extract unique names from the 'condensedName' column
    if 'condensedName' in filtered_ad.columns:
        unique_names = set(filtered_ad['condensedName'].dropna().unique())
    else:
        raise KeyError("'condensedName' column not found in the DataFrame")

    def filter_mentions(mentions):
        if isinstance(mentions, list):
            return [mention for mention in mentions if mention in unique_names]
        return []

    # Apply the filter function to the 'Mention' column
    filtered_ad['Mention'] = filtered_ad['Mention'].apply(filter_mentions)

    # Remove rows where 'Mention' column is an empty list
    filtered_ad = filtered_ad[filtered_ad['Mention'].apply(
        lambda x: len(x) > 0)]

    # Convert mentions back to comma-separated string
    filtered_ad['Mention'] = filtered_ad['Mention'].apply(
        lambda x: ', '.join(x))

    return filtered_ad


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""Function to calculate 4 different Centrality"""


def norm_centrality_key(key):
    return re.sub(r'\W+', '', key).lower()


def centrality_dupliCheck(centrality):
    NEW_CENTRALITY = {}
    final_centrality = {}
    normalized_to_original = {}

    numeric_key_pattern = re.compile(r'^\d+(, \d+)*$')

    for concatenated_keys, value in centrality.items():
        if numeric_key_pattern.match(concatenated_keys):
            individual_keys = concatenated_keys.split(', ')
            for key in individual_keys:
                normalized_key = norm_centrality_key(key)
                if normalized_key in normalized_to_original:
                    original_key = normalized_to_original[normalized_key]
                    final_centrality[original_key] = round(
                        final_centrality[original_key] + value, 2)
                else:
                    normalized_to_original[normalized_key] = key
                    final_centrality[key] = round(value, 2)
                NEW_CENTRALITY[key] = round(value, 2)
        else:
            normalized_key = norm_centrality_key(concatenated_keys)
            if normalized_key in normalized_to_original:
                original_key = normalized_to_original[normalized_key]
                final_centrality[original_key] = round(
                    final_centrality[original_key] + value, 2)
            else:
                normalized_to_original[normalized_key] = concatenated_keys
                final_centrality[concatenated_keys] = round(value, 2)
            NEW_CENTRALITY[concatenated_keys] = round(value, 2)

    return final_centrality


def get_max_centrality(degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality):

    # Get the node (user) with the maximum degree centrality
    max_degree_node = max(degree_centrality, key=degree_centrality.get)
    # #print(f"Node with Maximum Degree Centrality:{max_degree_node} \
    #     [{degree_centrality[max_degree_node]}]") # %tage of direct communication

    # Get the node (user) with the maximum closeness centrality
    max_closeness_node = max(closeness_centrality,
                             key=closeness_centrality.get)
    # #print(f"Node with Maximum closeness Centrality: {max_closeness_node} \
    #     [{closeness_centrality[max_closeness_node]}]") # High closeness -> eff. info. spread and influence

    # Get the node (user) with the maximum betweenness centrality
    max_betweenness_node = max(
        betweenness_centrality, key=betweenness_centrality.get)
    # #print(f"Node with Maximum betweenness Centrality: {max_betweenness_node} \
    #     [{betweenness_centrality[max_betweenness_node]}]") # no intermediary bridge

    # Get the node (user) with the maximum eigenvector centrality
    max_eigenvector_node = max(
        eigenvector_centrality, key=eigenvector_centrality.get)
    # #print(f"Node with Maximum eigenvector Centrality: {max_eigenvector_node} \
    #     [{eigenvector_centrality[max_eigenvector_node]}]") # most influential node - has significant influence on others


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""Key influencers check"""


def key_influencer_inGroup(Ga, communities):

    # Initialize a dictionary to store key influencers for each community
    key_influencers = {}

    # Iterate over each community
    for i, community in enumerate(communities, start=1):
        # Create a subgraph for the community
        community_graph = Ga.subgraph(community)

        # Calculate degree centrality for nodes in the community
        degree_centrality = nx.degree_centrality(community_graph)

        # Sort nodes by degree centrality
        sorted_nodes = sorted(degree_centrality.items(),
                              key=lambda x: x[1], reverse=True)

        # Store the top influencers for the community
        # Adjust the number of influencers as needed
        key_influencers[f"Community {i}"] = sorted_nodes[:3]

    # #Print the top influencers for each community
    for community, influencers in key_influencers.items():
        # #print(f"{community} - Top Influencers:\n")
        for influencer, centrality in influencers:
            # #print(f"\t{influencer}: {centrality:.4f}")
            pass
        #print('\n')


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""Overlapping users in communities"""

# Function to normalize user names


def normalize_username(username):
    return re.sub(r'\W+', '', username).lower()

# Function to check if a username has both special characters and whitespaces


def has_special_chars_and_whitespaces(username):
    has_special_chars = re.search(r'\W', username) is not None
    has_whitespaces = re.search(r'\s', username) is not None
    return has_special_chars and has_whitespaces


def overlapping_users(communities):
    # Initialize a dictionary to track the communities each normalized user belongs to
    user_communities = defaultdict(set)
    original_usernames = defaultdict(set)

    # Iterate over each community
    for i, community in enumerate(communities, start=1):
        community_label = f'Community {i}'
        for user in community:
            normalized_user = normalize_username(user)
            user_communities[normalized_user].add(community_label)
            original_usernames[normalized_user].add(user)

    # Find users who belong to more than one community
    multi_community_users = {user: comms for user,
                             comms in user_communities.items() if len(comms) > 1}

    # Compile the final data into a dictionary
    compiled_data = {}

    # #Print the users and the communities they belong to, along with original usernames
    #print("Users in multiple communities:")
    for user, comms in multi_community_users.items():
        original_names = [name for name in original_usernames[user]
                          if has_special_chars_and_whitespaces(name)]
        original_names_str = ', '.join(original_names)
        # #print(f"Original Names: {original_names_str}, \tCommunities: {', '.join(comms)}")
        compiled_data[original_names_str] = list(comms)

    # #Print the compiled data
    # #print("\nCompiled Data:", len(compiled_data))
    if (compiled_data):
        return compiled_data


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""


def config_emoji_paths():
    # Find the Symbola font path
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    symbola_font_path = None

    for path in font_paths:
        if 'Symbola' in path:
            symbola_font_path = path
            break

    # Check if the font path was found
    if not symbola_font_path:
        raise FileNotFoundError(
            "Symbola font not found. Please ensure it is installed on your system.")
    else:
        print(symbola_font_path)

    # Path to the installed fonts
    # Path to Symbola font
    symbola_font_path = r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\Symbola.otf"
    # Path to Noto Emoji font
    noto_emoji_font_path = r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\NotoEmoji.ttf"

    # Set up FontProperties for each font
    symbola_prop = fm.FontProperties(fname=symbola_font_path)
    noto_emoji_prop = fm.FontProperties(fname=noto_emoji_font_path)

    # Register the fonts with Matplotlib
    fm.fontManager.addfont(symbola_font_path)
    fm.fontManager.addfont(noto_emoji_font_path)

    # Configure Matplotlib to use the desired fonts
    # Use Symbola for regular text
    plt.rcParams['font.family'] = symbola_prop.get_name()
    # Use both Symbola and Noto Emoji for sans-serif font
    plt.rcParams['font.sans-serif'] = [symbola_prop.get_name(),
                                       noto_emoji_prop.get_name()]


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""Function to check if a character is likely an emoji based on Unicode properties"""
# Function to check if a character is likely an emoji based on Unicode properties


def is_emoji(char):
    """
    This function checks if a character is likely an emoji based on Unicode properties.
    Args:
        char: A single character to be evaluated.
    Returns:
        True if the character's code point falls within an emoji range, False otherwise.
    """
    # Emoji code point ranges (adjust if needed)
    emoji_ranges = [
        (0x1F600, 0x1F64F), (0x1F680, 0x1F6C5), (0x1F300, 0x1F5FF),
        (0x2600, 0x26FF), (0x2700, 0x27BF), (0xFE00, 0xFE0F)
    ]
    # Convert character to its Unicode code point
    code_point = ord(char)
    # Check if the code point falls within an emoji range
    for start, end in emoji_ranges:
        if start <= code_point <= end:
            return True
    return False

# Function to count emojis


def count_emojis(message):
    emoji_counts = {}
    for char in message:
        if is_emoji(char):
            emoji_counts[char] = emoji_counts.get(char, 0) + 1
    # #print("\n count_emojis Passed!\n")
    return emoji_counts

# Function to calculate total emoji count from emoji counts dictionary


def total_emoji_count(emoji_counts):
    if not emoji_counts:  # Handle empty dictionary case
        return 0
    # #print("\n total_emoji_count Passed!\n")
    return sum(emoji_counts.values())

# Function to count occurrences of '<Media omitted>' in a message


def count_media_omitted(message):
    # #print("\n count_media_omitted Passed!\n")
    # return len(re.findall(r'<Media omitted>', message))
    if isinstance(message, str):
        return len(re.findall(r'<Media omitted>', message))
    return 0


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""top 5 emojis in the dataset:"""


def top5_emoji_count(Emoji_DF):
    # Count Individual Emojis over the entire DF

    ind_emo_count = {}

    # loop the DF to get the emoji column
    for index, row in Emoji_DF.iterrows():
        row_emoji_dict = row['Emoji_Counts']

        # Loop inside the emoji count column
        for key, value in row_emoji_dict.items():
            # Filter out emojis that are whitespace or have no visible character
            if key.strip() and ord(key) not in [0xFE0F, 0x1F3FB, 0x263A, 0x1F3FE, 0x267E, 0x1F3FF, 0x1F3FC, 0x1F3FD]:
                if key in ind_emo_count:
                    ind_emo_count[key] += value
                else:
                    ind_emo_count[key] = value
    # Get the top 5 emojis
    top_5_emojis = sorted(ind_emo_count.items(),
                          key=lambda item: item[1], reverse=True)[:5]

    # #Print the top 5 emojis
    #print("Total Distinct Emojis Found in Dataset: ", len(ind_emo_count.keys()))

    for emoji, count in top_5_emojis:
        #print(f"Emoji: {emoji} Count: {count}")
        unicode_point = ord(emoji)
        print(f"Emoji: {emoji} Count: {count} \tUnicode: 0x{unicode_point:04X}")

    return ind_emo_count


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

# Function to get the top emoji for a single person's emoji count dictionary


def get_top_emoji(emoji_counts):
    if emoji_counts:
        return max(emoji_counts.items(), key=lambda item: item[1])
    return (None, 0)


"""********************************************************************************************************************************"""
"""********************************************************************************************************************************"""

"""Main Builder Function"""


def main(filepath,BASE_DIR):

    # file path is upload path
    input_folder_path = os.path.dirname(filepath)
    output_folder_path = os.path.join(BASE_DIR, 'DATA', 'uploads')  
    """*
    *
    * To do this - configure correct directory paths
    *
    *"""
    # OUTPUT_FOLDER = os.path.join(BASE_DIR, 'DATA', 'Outputs')  

    txt_file_name = "parsed_text_file_raw.txt"
    # #print(os.path.join(output_folder_path,txt_file_name))
    parsed_txt_file_path = os.path.join(output_folder_path, txt_file_name)

    print(f"\ninput text file: {input_folder_path}")
    print(f"\nouput text file: {output_folder_path}")
    print(f"\nparsed_txt_file_path: {parsed_txt_file_path}")
    time.sleep(2)
    # input("\nenter........")
    try:
        pass
        """step - 1"""
        format_txt_file(input_folder_path, output_folder_path)

        """step - 2"""
        ParsedData = parse_raw_txt_file(parsed_txt_file_path)
        """ON RETURN check"""
        # check the parsed File
        LL = random.randint(0, len(ParsedData))
        #print("\n", ParsedData[:1])
        #print("\n", ParsedData[LL:LL+2])

        """step - 3"""
        """Get the DataFrame"""
        df = initial_preprocessing(ParsedData)

        """step - 4"""
        """Format Date-Time"""
        # Apply the custom date parsing function to the Date column
        df['Date'] = df['Date'].apply(parse_date)
        # Convert the Date column to datetime
        df['Date'] = pd.to_datetime(
            df['Date'], format='%d/%m/%Y', errors='coerce')
        # Apply the custom time parsing function to the Time column
        df['Time'] = df['Time'].apply(parse_time)
        # Convert the Time column to a datetime.time object
        df['Time'] = pd.to_datetime(
            df['Time'], format='%H:%M', errors='coerce').dt.time

        """step - 5"""

        """ Concatenate 'Date' and 'Time' columns into a new column 'Datetime' 
            For Future calender extraction """

        # Combine 'Date' and 'Time' into a single 'DateTime' column
        df['DateTime'] = df.apply(lambda row: datetime.combine(
            row['Date'], row['Time']), axis=1)
        df.reset_index(inplace=True)
        """Extract hour, day, day name, month and year"""
        df['hour'] = df['DateTime'].dt.hour
        df['day'] = df['DateTime'].dt.day
        df['day name'] = df['DateTime'].dt.day_name()
        df['month'] = df['DateTime'].dt.month
        df['year'] = df['DateTime'].dt.year

        # Check
        #print("\n\nAll Executed\n")
        time.sleep(2)

        """EDA"""

        """step - 6"""
        # 1. Message Frequency Analysis
        total_msg_per_day = Message_Frequency_Analysis(df)
        daily_msg_count_filtered = IQR(total_msg_per_day)

        # plot graphs:
        # print("\n\ndo the graphs now 1-4")
        # time.sleep(5)
        # input("\nEnter to Continue")
        if (len(daily_msg_count_filtered) > 0):

            graphs.daily_msg_plot_1(daily_msg_count_filtered)
            return True
            graphs.daily_msg_logT_plot_2(daily_msg_count_filtered)
            graphs.daily_msg_hist_plot_3(total_msg_per_day)
            graphs.daily_msg_box_plot_4(daily_msg_count_filtered)
        # print("\n\ndone with the graphs now 1-4")
        # time.sleep(5)
        # input("\nEnter to Continue")
        return True
        """step - 7"""
        # 2.
        hourly_msg_count, monthly_msg_count = hourly_FA(df)

        if (len(hourly_msg_count) > 0):

            # plot graphs
            graphs.hourly_barLine_plot_5(hourly_msg_count)
            graphs.hourly_msg_box_plot_6(hourly_msg_count)

        if (len(monthly_msg_count) > 0):

            graphs.monthly_barLine_plot_7(monthly_msg_count)
            graphs.monthly_msg_box_plot_8(monthly_msg_count)

        hour_month_stats(hourly_msg_count, monthly_msg_count)

        """step - 8"""

        """Most Active Users (in no of msg sent) List in entire DF"""
        most_active_users_list = df.groupby(df['Name'])['Message'].count()
        most_active_users_list.sort_values(ascending=False).head()

        if (len(most_active_users_list) > 0):
            graphs.most_active_user_plot_9(most_active_users_list)

        """Get Peak Messaging time per day and not the entire DF set"""
        peak_msg_time_per_day = df.groupby(
            ['Date', 'day', 'hour']).size().reset_index(name='message_count')
        peak_msg_time_per_day = peak_msg_time_per_day.loc[peak_msg_time_per_day.groupby(
            'day')['message_count'].idxmax()]

        #print("Peak message time per day:")
        #print(peak_msg_time_per_day[['day', 'hour', 'message_count']].head(3))

        # Extracting data for plotting
        tdays = peak_msg_time_per_day['day']
        thours = peak_msg_time_per_day['hour']
        tmessage_counts = peak_msg_time_per_day['message_count']

        if (len(tmessage_counts) > 0):

            graphs.peak_msg_ToD_plot_10(tdays, thours, tmessage_counts)
            graphs.interDH_scatter_plot_11(tdays, thours, tmessage_counts)

        """step - 9"""
        """Get Peak Messaging day per week."""

        week_df = week_analysis(df)

        if (len(week_df) > 0):
            # plot
            graphs.peak_msg_DPweek_plot_12(week_df)

        """step - 10"""
        """Segment Users Based on Activity Level"""

        "Average Day where Most Messages were shared for the aboce graph insight"
        avg_msg_day_general = df.groupby(df['most_messaged_day']).size()

        #print("Total No. of Messages sent each day of the week for the entire dataset:\n")
        #print(avg_msg_day_general.sort_values(ascending=False))
        # print("\nMost active day in the entire DF: ",avg_msg_day_general.idxmax())

        wde_data_list = user_segmentation(df, most_active_users_list)

        if (len(wde_data_list) > 0):

            graphs.wdwe_activity_level_plot_13(wde_data_list)

            # weekday / weekend top 5 plot:
            graphs.wdwe_top5_active_plot_14(wde_data_list)
            graphs.resampled_DWM_plot_15(df)

        """step - 11"""
        # 3. Message Content Analysis
        """Tokenisation of all the messages"""

        # not doing : Plot word cloud:
        # graphs.wordCloud_plot_17(word_freq)
        # try:
        #     df.to_csv(r'D:\_Daanish_files\VsCode_PY\Data_Science\otherChats\df.csv')
        # except:
        #     #print(Exception)
        # return True
        # exit(1)
        #print("\n\nNetwork Analysis\n\n")
        time.sleep(2)

        """Network Analysis"""

        """step - 12"""
        """### Extraction of @Mentions and Names alike from Messages to check
        ### how many users explicitly wants a particular user to check the message
        ### New DataFrame DFC Created from DF """

        # Create DataFrame with required columns
        # df = pd.read_csv(r'D:\_Daanish_files\VsCode_PY\Data_Science\otherChats\df.csv')

        dfc = pd.DataFrame(df)[['Name', 'Message']].copy()
        dfc['condensedName'] = dfc['Name'].str.replace(
            r'\W', '', regex=True).str.lower()  # Remove non-word characters and lowercase
        dfc['Mention'] = ''  # Create an empty "Mention" column

        test_dfc = extract_mentions(dfc)

        if (len(test_dfc) > 0):
            #print("\n success")
            time.sleep(2)
        else:
            return True
            exit(1)
        # return True
        # test_dfc.to_csv(r'D:\_Daanish_files\VsCode_PY\Data_Science\otherChats\test_dfc.csv')
        # test_dfc = pd.read_csv(r'D:\_Daanish_files\VsCode_PY\Data_Science\otherChats\test_dfc.csv')

        """step - 12"""
        """Created new DataFrame : 
        filtered_dfc from DFC for Network Analysis Ahead Analyzing Network Structure"""

        # Create a filtered DataFrame without modifying the original one

        filtered_ad = test_dfc.dropna(subset=['Mention'])

        filtered_ad = filtered_ad[filtered_ad['Mention'].astype(
            str).str.strip() != '']

        filtered_ad = filtered_ad.drop(filtered_ad[filtered_ad['Mention'].str.len()
                                                   < filtered_ad['Name'].str.len().min()].index)

        """One Last Filtering of Mentions"""
        filtered_ad = final_pure_mentions(filtered_ad)
        time.sleep(2)
        # filtered_ad.to_csv(r'D:\_Daanish_files\VsCode_PY\Data_Science\otherChats\filtered_ad.csv')

        # Create a directed graph from the interaction data in dfc
        Ga = nx.from_pandas_edgelist(
            filtered_ad, source='Name', target='Mention', create_using=nx.DiGraph())

        # Basic network analysis
        #print("Number of nodes (users):", Ga.number_of_nodes())
        #print("Number of edges (interactions):", Ga.number_of_edges())

        """Has wrong names with spl.charc."""

        # plot graphs:
        graphs.basic_graph_spring_plot_16(Ga)

        """step - 12"""

        # Calculate degree centrality
        degree_centrality = centrality_dupliCheck(nx.degree_centrality(Ga))

        # Calculate betweenness centrality
        betweenness_centrality = centrality_dupliCheck(
            nx.betweenness_centrality(Ga))

        # Calculate closeness centrality
        closeness_centrality = centrality_dupliCheck(
            nx.closeness_centrality(Ga))

        # Calculate eigenvector centrality with increased maximum iterations
        eigenvector_centrality = centrality_dupliCheck(
            nx.eigenvector_centrality(Ga, max_iter=1000))

        # #print max centrality
        get_max_centrality(degree_centrality, betweenness_centrality,
                           closeness_centrality, eigenvector_centrality)

        time.sleep(2)
        """step - 12"""
        graphs.grouped_centrality_line_plot_18(
            degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality)

        """step - 12"""
        # Community Finidings
        # Detect communities using the louvain_communities algorithm
        communities = nx.community.louvain_communities(Ga)
        # Convert communities to a list of sets for easier handling
        communities = list(communities)
        community_sizes = [len(community) for community in communities]
        community_labels = [
            f'Community {i+1}' for i in range(len(communities))]

        time.sleep(2)

        if (len(community_sizes) > 0):
            graphs.user_per_Community_plot_19(
                community_sizes, community_labels)
        if (len(communities) > 0):
            graphs.user_per_Community_graph_plot_20(Ga, communities)

        # creating mapping on which user belongs to which community number from above
        community_indices = {node: i for i,
                             comm in enumerate(communities) for node in comm}
        # #print(community_indices)
        #print("\n\n")
        if (len(community_indices) > 0):
            graphs.inter_community_heatmap_plot_21(
                Ga, community_indices, communities)
            print('\n')

        else:
            print("community indices error")

        # going for key influencers:
        key_influencer_inGroup(Ga, communities)

        # user in multiple community:
        compiled_data = overlapping_users(communities)
        if compiled_data:
            graphs.user_intersect_comm_plot_22(compiled_data)
            #print('\n')

        # time.sleep(5)

        """Emoji and Media"""

        """step - 12"""
        #print("\n\n doing emoji analysis now below\n")
        # config emoji paths - not required now
        time.sleep(2)

        """step - 13"""
        # another sub df for emoji:
        temp_emo_df = dfc[['Name', 'Message']]  # Select relevant columns
        # #print("\n\n Select relevant columns in temp_emo_df\n", temp_emo_df.dtypes)
        # time.sleep(2)

        # Wait for user input to continue
        # input("\npre-try: Press Enter to continue...\n")

        # Convert all messages to strings and handle NaNs
        temp_emo_df['Message'] = temp_emo_df['Message'].astype(str).fillna('')
        # Group messages by name and concatenate them into a single message string per user
        Emoji_DF = temp_emo_df.groupby('Name')['Message'].apply(
            lambda msg: '\n'.join(msg)).reset_index()

        # # Wait for user input to continue
        # input("\nexcept: Press Enter to continue...\n")

        os.system("CLS")
        # try:

        # Add new column for emoji counts
        Emoji_DF['Emoji_Counts'] = Emoji_DF['Message'].apply(count_emojis)

        # Add a new column for total emoji count
        Emoji_DF['Total_Emojis_Count'] = Emoji_DF['Emoji_Counts'].apply(
            total_emoji_count)

        # MediaValues = temp_emo_df[temp_emo_df['Message']
        #                           == '<Media omitted>'].reset_index()

        # Convert all messages to strings and handle NaNs
        df['Message'] = df['Message'].astype(str).fillna('')

        # Apply the function to the 'Message' column and store the result in a new column
        df['MediaCount'] = df['Message'].apply(count_media_omitted)

        # Group by the 'Name' column and sum the 'MediaCount' to get the total count per user
        media_counts = df.groupby('Name')['MediaCount'].sum().reset_index()

        # Merge media_counts with Emoji_DF and rename the merged column in one step
        Emoji_DF = Emoji_DF.merge(media_counts, on='Name', how='left').rename(
            columns={'MediaCount_media': 'MediaCount'})

        # Fill NaN values in MediaCount with 0 (for users who didn't send any media messages)
        Emoji_DF['MediaCount'] = Emoji_DF['MediaCount'].fillna(0)

        if (len(Emoji_DF) > 0):

            # plot media and emoji
            graphs.Total_media_per_user_plot_23(Emoji_DF)

            graphs.Total_emoji_per_user_plot_24(Emoji_DF)
        # time.sleep(2)

        """step - 15"""

        # Count Individual Emojis over the entire DF
        ind_emo_count = top5_emoji_count(Emoji_DF)

        # Filter the overall emoji counts to include only those with a count greater than 10
        filtered_emoji_counts = {emoji: count for emoji,
                                 count in ind_emo_count.items() if count > 15}

        # Prepare the data for plotting
        # emojis = list(filtered_emoji_counts.keys())
        # counts = list(filtered_emoji_counts.values())
        if (len(filtered_emoji_counts) > 0):
            graphs.count_distinct_emojis_plot_25(filtered_emoji_counts)

        """step - 16"""

        # List to store the results
        top_emojis_per_person = []

        # Loop through the DataFrame to get each person's top emoji
        for index, row in Emoji_DF.iterrows():
            name = row['Name']
            top_emoji, count = get_top_emoji(row['Emoji_Counts'])
            top_emojis_per_person.append((name, top_emoji, count))

        # #Print the results
        # for item in top_emojis_per_person:
        #     #print(f"Name: {item[0]}, Top Emoji: {item[1]}, Count: {item[2]}")
        # top_emojis_per_person[:2]

        if (len(top_emojis_per_person) > 0):
            graphs.top_emoji_per_user_plot_26(top_emojis_per_person)

        """step - 17"""

    except Exception as error:
        print(f"Error: {error}")

    finally:
        return True
