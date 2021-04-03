"""
Author: John McDonald
This file contains all functionality for managing COPIS Database records.
"""

import argparse
import csv
import pandas as pd
import re
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def addAttribute(attribute, type):
    """
    Used in Add Mode to prompt user and check format of inputs
    :param attribute: name of column
    :param type:
    :return:
    """
    while True:
        ans = input("Enter Attribute [{0}] of Type [{1}]: ".format(attribute, type))

        # Enter default values
        if ans == "":
            if attribute == 'Tdt': return datetime.date.today().strftime("%B, %d, 20%y")
            elif attribute in ['PStatus', 'IStatus']: return 'U'
            elif attribute in ['Gen', 'Email', 'History', 'Notes']: return ''
            else: print("Input required.")

        # Check format of attribute entry
        if validFormat(attribute, ans): return ans
        else: print("Invalid answer.")


def validFormat(attribute, ans):
    """
    Takes an atribute, and a value, and checks whether the format is valid based on COPIS DB format requirements.
    :param attribute: attribute to check
    :param ans: value to check
    :return: Boolean, True if valid format, False if invalid format.
    """
    # Regex formulas
    reg_Tdt = '^(January|February|March|April|May|June|July|August|September|October|November|December)+,+\\s+(\\d{1,2})+,+\\s+(\\d{4})$'
    reg_date = '(January|February|March|April|May|June|July|August|September|October|November|December)+,+\\s+(\\d{1,2})+,+\\s+(\\d{4})'
    reg_status = '(P|N|U|Q|H|I|C|R|D)'
    reg_History = '(' + reg_status + '+:+' + reg_date + ')+(,+\\s' + reg_status + '+:+' + reg_date + ')*'

    # TODO county only alphabet letters
    if attribute == "Pno" and ans.isdigit():
        return True
    elif attribute == "Age" and re.match("^\d+(\.\d+)?$", ans):
        return True
    elif attribute == "Gen" and re.match('^[M|F]$', ans):
        return True
    elif attribute == "County" and re.match('^\w+(\s\w+)*$', ans):
        return True
    elif attribute == "IStatus" and re.match('^[U|N|P]$', ans):
        return True
    elif attribute == "PStatus" and re.match('^[U|Q|H|I|C|R|D]$', ans):
        return True
    elif attribute == "Tdt" and re.match(reg_Tdt, ans):
        return True
    elif attribute == "Email" and re.match(r"[^@]+@[^@]+\.[^@]+", ans):
        return True
    elif attribute == "History" and re.match(reg_History, ans):
        return True
    elif attribute == "Notes":
        return True
    elif attribute in ['Gen', 'Tdt', 'Email', 'History', 'Notes'] and ans == '':
        return True
    else:
        return False


def promptSearchCriteria():
    """ Prompt user for search criteria.  Useful in both change and search modes.
    :return: panda dataframe of search results
    """
    # Prompt user until correct search input
    while True:
        search_string = input("\nEnter search criteria with value [<attribute>:<value>,...] : ")

        attributes = re.findall('\w+:', search_string)
        attributes = [x[:-1] for x in attributes]
        values = re.split(',?\s?[a-zA-Z]{2,}:\s?', search_string)[1:]

        #  Split input into -> [['column','value'],['column2','value2],...]
        split_search = list(zip(attributes, values))

        # Check input for formatting
        if len(split_search) == 0:
            print("Search criteria not entered. Help text about the program:")
            parser.print_help()
            print("Exiting program...")
            exit(-1)
        elif any([value == '' for attribute, value in split_search]):
            print('Incorrect format; value not entered.')
        elif all([validFormat(attribute, value) for attribute, value in split_search]):
            print('Searching for criteria...')
            break
        else:
            print('Incorrect format for value and attribute.')

    # Search each row for all attributes
    rows_found = []
    with open(fname, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            if all([values in row[column] for column, values in split_search]): rows_found.append(row)

    # Convert to dataframe for printing
    return pd.DataFrame(rows_found)


def removeInvalidFormat(df):
    """
    Takes a dataframe, and removes all columns with rows that have invalid format.
    :param df: Input dataframe
    :return: Refined dataframe with removed rows of invalid format
    """
    indexs_to_remove = set([])
    attributes_missing = []
    for i, row in df.iterrows():
        for attribute, value in row.iteritems():
            value = str(value)
            if not validFormat(attribute, value):
                attributes_missing.append(attribute)
                indexs_to_remove.add(i)

    print("Removing {} records".format(len(indexs_to_remove)))

    uniques = list(dict.fromkeys(attributes_missing))
    for unique in uniques:
        print("{0} Records missing {1}".format(attributes_missing.count(unique), unique))
    print()

    return df.drop(indexs_to_remove)


def sortPno(df_update):
    """
    Sorts a dataframe based on Pno value, provided the column is of string datatype.
    :param df_update: Dataframe to sort
    :return: Sorted dataframe based on Pno column
    """
    df_update.Pno = df_update.Pno.astype(int)
    df_update = df_update.sort_values(by=['Pno'])
    df_update.Pno = df_update.Pno.astype(str)
    return df_update


def tallyCounter(l):
    """
    Takes a list and returns a dict of unique keys, with values for number of occurances
    :param l: Input list
    :return: Tallied Dictionary
    """
    d = {}
    for i in l:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return d


def countRangeInList(li, min, max):
    """
    Counts the number of occurances of values in a list, between specified minimum and maximum values.
    :param li: list of values
    :param min: minimum for range
    :param max: maximum for range
    :return: number of values in list, within specified range
    """
    ctr = 0
    for x in li:
        if min <= x <= max:
            ctr += 1
    return ctr

parser = argparse.ArgumentParser(description='Management application for manipulating data in COPIS')
parser.add_argument("-m", type=str, nargs='+', help="Mode name for operations. Mode options are: search, change, add <no>, import <FiletoImport>.")
parser.add_argument("-g", choices=["day", "age", "gen", "county"], type=str,
                    help="Print graph of data provided argument.")
parser.add_argument("-f", default="coviddata.csv", type=str, help="File name to be scanned, changed, analyzed, etc.")
parser.add_argument("-i", action='store_true', help="Print average information for all categories.")

# arguments
args = parser.parse_args()
mode = args.m
fname = args.f
i = args.i
g = args.g

# Check if passed filename exists
if not os.path.exists(fname):
    print("WARNING: Given file name for -f does not exist. Exiting program.")
    exit(-1)

# Check format of file given
try:
    df = pd.read_csv(fname, delimiter='\t', dtype=str)
    # Accepted field names
    fieldnames = ['Pno', 'Age', 'Gen', 'County', 'IStatus', 'PStatus',
                  'Tdt', 'Email', 'History', 'Notes']
except:
    print('Could not read csv file.  Please check file format for COPIS standards.')
    exit(-1)

if df.columns.values.tolist() != fieldnames:
    print("Field names do not follow COPIS Standards. Please check formatting.")
    exit(-1)

if mode != None:

    if mode[0] == 'search':

        # Dataframe of search results
        rows_found = promptSearchCriteria()

        # TODO enter 10 at a time
        if len(rows_found) > 10:
            ans = input("More than 10 records matching the criteria, print (y/n): ")
            if (ans == 'y'):
                print('\n' + rows_found.to_string())
            elif (ans == 'n'):
                print('Exiting program...')
                exit(-1)
            else:
                print('Command not recognized. Exiting program...')
                exit(-1)
        elif len(rows_found) == 0:
            print('No records found for search criteria.')
        else:
            print('Found {0} records:\n'.format(len(rows_found)))
            print(rows_found.to_string())

    elif mode[0] == 'add':

        if len(mode) == 1:
            no_records = 1
        elif mode[1].isdigit():
            no_records = int(mode[1])
        else:
            print(
                'Invalid argument entry. Add mode should be followed by number of records to add; otherwise, defaults to 1.')
            exit(-1)

        # Read file provided to dataframe
        df_import = pd.read_csv(fname, delimiter='\t', dtype=str)

        # Filter out anything besides int
        column_Pno = [int(x) for x in df_import["Pno"].tolist() if x.isdigit()]
        # Store max Pno to calculate new record Pno
        max_Pno = max(column_Pno)

        # ['Age','Gen','County','IStatus','PStatus','Tdt','Email','History','Notes']
        attributes = list(df_import.columns[1:])
        # Used for UI with above attributes
        types = ['Float*', 'String {M,F}', 'String*', 'String {U:Unconfirmed, N:Negative, P:Positive}',
                 'String {U:Unknown, Q:Quarantined, H:Hospitalized, I:ICU, R:Recovered, D:Deceased}',
                 'Date as {Month, Day, Year}', 'String as {username@domain}', 'String as {PStatus:Tdt,...}', 'String']

        # List of List to hold all records, will save to csv file
        records = []
        counter = 1

        print("\nAdding {0} records to {1}".format(no_records, fname))
        # Loop until records (specified by user) are entered by user
        while counter <= no_records:
            print("\nPlease enter attributes for record {}. Types with * are required.".format(counter))
            record = ['']
            # Loop for each attribute in the record
            for attribute, type in zip(attributes, types):
                record.append(addAttribute(attribute, type))
            # Calculate max PNO for entered record and save to records
            max_Pno += 1
            record[0] = str(max_Pno)
            records.append(record)
            counter += 1

        # Save records to file
        with open(fname, 'a', newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            writer.writerows(records)

        print("\nRecords saved.")

    elif mode[0] == 'change':

        # Dataframe of search results
        rows_found = promptSearchCriteria()

        print("Record matching search criteria:\n")
        print(rows_found.to_string() + '\n')

        # Warn user of modifying all records
        if(len(rows_found) > 1):
            ans = input("Warning: all {0} matching records will be modified. Proceed? (y/n): ".format(len(rows_found)))
            if ans != "y":
                print("Exiting program...")
                exit(-1)

        # Keep prompting user until acceptable attribute recognized
        while True:
            attribute = input("Enter attribute to change: ")
            try:
                rows_found[attribute].tolist();
                if attribute in ['Pno','Tdt']:
                    print("Cannot change Pno or Tdt..")
                    continue
                elif attribute == 'e':
                    print("Exiting program...")
                    exit(-1)
                break
            except: print("Attribute not found.")

        print("Existing column values for {0}: ".format(attribute))
        print(rows_found[attribute].tolist())

        # Keep entering values until valid value recognized
        while True:
            value = input("Enter new value: ")
            if validFormat(attribute, value):
                rows_found[attribute].values[:] = value
                # Update history if status changed
                if attribute in ['IStatus','PStatus']:
                    rows_found['History'] = rows_found['History'].astype(str) + ', ' + value + ':' + datetime.date.today().strftime("%B, %d, 20%y")
                break
            else: print("Invalid format.")

        print("Records changed.")

        # Concatenate to existing file, sort, and save.
        df_update = pd.read_csv(fname, delimiter='\t', keep_default_na=False, dtype=str)
        df_update = pd.concat([df_update, rows_found]).drop_duplicates(['Pno'], keep='last')
        df_update = sortPno(df_update)
        df_update = df_update.reset_index(drop=True)
        df_update.to_csv(fname, sep='\t', index=False)

    elif mode[0] == 'import':

        # Check import mode has only 2 args
        if len(mode) != 2:
            print('Invalid argument entry. Import requires 1 additional argument with file to be imported name.')
            exit(-1)
        # check file to import exists
        elif not os.path.exists(mode[1]):
            print("WARNING: File to import name does not exist")
            exit(-1)

        fileToImport = mode[1]

        # dataframes
        df_import = removeInvalidFormat(pd.read_csv(fileToImport, delimiter='\t', keep_default_na=False, dtype=str))
        df_update = pd.read_csv(fname, delimiter='\t', keep_default_na=False, dtype=str)

        # Track PNO to modify and to add, each a list of integers
        # Modify, if PNO is in both df_import and df_update
        # Add, if PNO is in df_import but not in df_update
        pno_to_modify = list(set(df_import['Pno'].tolist()) & set(list(map(str, df_update['Pno'].tolist()))))
        pno_to_add = [x for x in df_import['Pno'].tolist() if x not in pno_to_modify]

        # Modify if user says 'y'
        if len(pno_to_modify) > 0:
            ans = input("Warning: File To Import contains {0} duplicate Pno's. Update existing records (y/n)?".format(len(pno_to_modify)))
            if (ans == 'y'):

                # update rows will be modified ; import rows contain for extraction
                rows_to_update = df_update.loc[df_update['Pno'].isin(pno_to_modify)].values.tolist()
                rows_to_import = df_import.loc[df_import['Pno'].isin(pno_to_modify)].values.tolist()

                # Append history, replace other columns
                for row_update, row_import in zip(rows_to_update, rows_to_import):
                    row_update[1:7] = row_import[1:7]
                    row_update[8] += ', ' + row_import[8]
                    row_update[9] = row_import[9]

                # Update dataframe with modified rows in 'rows_to_update'
                for pno in rows_to_update:
                    df_update.loc[df_update['Pno'] == pno[0], ['Age', 'Gen', 'County', 'IStatus', 'PStatus',
                                                               'Tdt', 'Email', 'History', 'Notes']] = pno[1:10]

                print('Modified {0} existing rows.'.format(len(pno_to_modify)))

            else:
                print("Modifying existing records cancelled.")

        # add
        for pno in pno_to_add:
            entry = df_import.loc[df_import['Pno'] == pno]
            df_update = pd.concat([df_update, entry], ignore_index=True)

        print('Added {0} rows.'.format(len(pno_to_add)))


        # sort and save
        df_update = sortPno(df_update)
        df_update = df_update.reset_index(drop=True)
        #print(df_update.to_string())
        df_update.to_csv(fname, sep='\t', index=False)

    else: print("Mode not recognized.  Please choose between search, add, change, import")

if i is True:

    print("Average Information mode entered for {0}".format(fname))
    print("First checking file for records with incorrect format.  Scanning file...")

    df = removeInvalidFormat(pd.read_csv(fname, delimiter='\t', keep_default_na=False, dtype=str))

    print("Printing average information for {0}:".format(fname))

    # convert columns to integer values
    df[["Pno", "Age"]] = df[["Pno", "Age"]].apply(pd.to_numeric)

    no_tested = df["Pno"].count()
    df_IStatus = df['IStatus'].value_counts()
    df_PStatus = df['PStatus'].value_counts()

    try: no_deceased = df_PStatus['D']
    except KeyError: no_deceased = 0
    try: no_confirmed = df_IStatus['P']
    except KeyError: no_confirmed = 0

    try: mortality_rate = round(no_deceased / no_confirmed, 2)
    except ZeroDivisionError: mortality_rate = 0

    df_IStatus_age = round(df[["IStatus", "Age"]].groupby("IStatus").mean(), 2)
    df_PStatus_age = round(df[["PStatus", "Age"]].groupby("PStatus").mean(), 2)

    try: avg_confirmed_age = df_IStatus_age["Age"]["P"]
    except KeyError: avg_confirmed_age = 0
    try: avg_deceased_age = df_PStatus_age["Age"]["D"]
    except KeyError: avg_deceased_age = 0

    df_IStatus_Gen = df[["IStatus", "Gen"]]
    count_IStatus_Gen = df_IStatus_Gen.loc[df['IStatus'] == 'P']['Gen'].value_counts()
    try: no_males = count_IStatus_Gen['M']
    except KeyError: no_males = 0
    try: no_females = count_IStatus_Gen['F']
    except: no_females = 0

    perc_male = round(no_males / (no_males + no_females), 2)
    perc_female = round(1.0-perc_male, 2)

    print("Total number of people tested: {0}".format(no_tested))
    print("Total confirmed cases: {0}".format(no_confirmed))
    print("Total deaths: {0}".format(no_deceased))
    print("Current mortality rate: : {0}".format(mortality_rate))
    print("Average age: : {0}".format(avg_confirmed_age))
    print("Average deceased age: : {0}".format(avg_deceased_age))
    print("%Female: {0} & %Male: {1}".format(perc_female, perc_male))

if g == "day":

    reg_confirmed = '(P:(?:January|February|March|April|May|June|July|August|September|October|November|December),\\s\\d{1,2},\\s\\d{4})'
    df = pd.read_csv(fname, delimiter='\t', keep_default_na=False)

    # DF with confirmed in history
    mask_confirmed = df.History.str.contains('P:')
    df_confirmed = df[mask_confirmed]

    # extract all confirmed dates from history:
    history_list = df_confirmed['History'].tolist()
    history_string = ' '.join(history_list)
    P_dates = re.findall(reg_confirmed,history_string)

    # Tally up confirmed cases for same date
    tally_confirmed = tallyCounter(P_dates)

    # create array of dates using key's from tally'd dictionary
    x = [d[2:] for d in tally_confirmed.keys()]
    x_values = [datetime.datetime.strptime(d, "%B, %d, 20%y").date() for d in x]

    # Create bounds for x_values array
    delta = max(x_values) - min(x_values)

    # Pad the dates array for all dates between minimum and maximum
    y_new = list(tally_confirmed.values())
    for i in range(delta.days + 1):
        day = min(x_values) + datetime.timedelta(days=i)
        if day not in x_values:
            x_values.append(day)
            y_new.append(0)

    x_values, y_new = (list(t) for t in zip(*sorted(zip(x_values, y_new))))

    y_total = np.cumsum(y_new)

    ax = plt.subplot()
    # Plot
    ax.plot(x_values, y_total)
    ax.plot(x_values, y_new)

    # Format x axis dates correctly, post biweekly
    formatter = mdates.DateFormatter("%b-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.WeekdayLocator()
    ax.xaxis.set_major_locator(locator)
    plt.xticks(rotation=90)
    ax.set_title('Cases by Day')

    plt.show()

elif g == 'age':

    df = pd.read_csv(fname, delimiter='\t', keep_default_na=False)
    ranges = [[0,4],[5,9],[10,14],[15,17],[18,24],[25,34],[35,44],[45,54],[55,64],[65,74],[75,84],[85,120]]

    # DF P in history
    mask_confirmed = df.History.str.contains('P:')
    df_confirmed = df[mask_confirmed]
    # list of ages
    ages_confirmed = df_confirmed['Age'].tolist()
    # count ages in range
    y_confirmed = []
    for low, high in ranges:
        y_confirmed.append(countRangeInList(ages_confirmed,low,high))


    # DF D in history
    mask_deceased = df.History.str.contains('D:')
    df_deceased = df[mask_deceased]
    # list of ages
    ages_deceased = df_deceased['Age'].tolist()
    # count ages in range
    y_deceased = []
    for low, high in ranges:
        y_deceased.append(countRangeInList(ages_deceased,low,high))

    x = np.arange(len(y_confirmed))

    # duel bar graph

    x_labels = [str(l[0]) + '-' + str(l[1]) for l in ranges]
    x_labels[11] = '85>'

    ax = plt.subplot(111)
    w = 0.3
    ax.bar(x - (w/2), y_confirmed, width=w, color='b', align='center', label='Total cases')
    ax.bar(x + (w/2), y_deceased, width=w, color='g', align='center', label='Deaths')
    ax.autoscale(tight=True)

    for i, v in enumerate(y_confirmed):
        ax.text(i - (w/2), v + .1, str(v), color='b', fontsize=6)

    for i, v in enumerate(y_deceased):
        ax.text(i + (w/2), v + .1, str(v), color='g', fontsize=6)

    plt.xticks(x, x_labels)
    plt.legend()
    ax.set_title('Cases by Age')
    ax.grid(axis='y')
    plt.show()

elif g == 'gen':

    df = pd.read_csv(fname, delimiter='\t', keep_default_na=False)
    ranges = [[0,4],[5,9],[10,14],[15,17],[18,24],[25,34],[35,44],[45,54],[55,64],[65,74],[75,84],[85,120]]

    # DF P in history
    mask_confirmed = df.History.str.contains('P:')
    df_confirmed = df[mask_confirmed]

    # list of both genders, ages [[M, 14],..]
    gen = df_confirmed['Gen'].tolist()
    age = df_confirmed['Age'].tolist()
    gen_age = [list(x) for x in zip(gen, age)]

    # list of single genders, ages [gen, age]
    m = [x for x in gen_age if x[0] == 'M']
    f = [x for x in gen_age if x[0] == 'F']

    # list of ages only
    m_ages = [x[1] for x in m]
    f_ages = [x[1] for x in f]

    y_values_male = []
    y_values_female = []

    # count ages in range
    for low, high in ranges:
        y_values_male.append(countRangeInList(m_ages,low,high))
        y_values_female.append(countRangeInList(f_ages,low,high))

    # Plot graph

    x_labels = [str(l[0]) + ' to ' + str(l[1]) + ' Years' for l in ranges]
    x_labels[11] = '85>'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(y_values_male))
    y_values_female = [-x for x in y_values_female]

    w = 0.2
    ax.barh(x, y_values_male, height=w, color='b', align='center', label='male')
    ax.barh(x, y_values_female, height=w, color='g', align='center', label='female')

    x = list(x)

    # Print tick values next to each bar
    for i, v in enumerate(y_values_male):
        ax.text(v/2, i+.25, str(v), color='b', fontsize=6)

    for i, v in enumerate(y_values_female):
        ax.text(v/2, i+.25, str(abs(v)), color='g', fontsize=6)

    ax.axvline(0, color='k', lw=1)
    plt.yticks(x, x_labels)
    plt.legend()
    plt.tick_params(bottom=False, labelbottom=False)
    ax.set_title('Confirmed Cases by Age & Gender')
    ax.grid(axis='x')
    plt.show()

elif g == 'county':

    df = pd.read_csv(fname, delimiter='\t', keep_default_na=False)

    # DF with confirmed in history
    mask_confirmed = df.History.str.contains('P:')
    df_confirmed = df[mask_confirmed]

    # DF with deceased in history
    mask_deceased = df.History.str.contains('D:')
    df_deceased = df[mask_deceased]

    # list of counties
    counties_confirmed = df_confirmed['County'].tolist()
    counties_deceased = df_deceased['County'].tolist()

    # Tally Dict
    counties_confirmed = tallyCounter(counties_confirmed)
    counties_deceased = tallyCounter(counties_deceased)

    # sort
    counties_confirmed = dict(sorted(counties_confirmed.items(), key=lambda x: x[1]))

    # declare lists for total counties, prep merge
    x_values = list(counties_confirmed.keys())
    y_values_confirmed = list(counties_confirmed.values())
    y_values_deceased = [0] * len(counties_confirmed)

    # merge
    for county, value in counties_deceased.items():
        if county in x_values:
            index = x_values.index(county)
            y_values_deceased[index] = value
        else:
            x_values.insert(0,county)
            y_values_confirmed.insert(0,0)
            y_values_deceased.insert(0,value)

    # Top 20 counties
    if len(y_values_confirmed) > 20:
        y_values_confirmed = y_values_confirmed[-20:]
        y_values_deceased = y_values_deceased[-20:]
        x_values = x_values[-20:]

    # plot horizontal graph

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # numbered array to plot counties list
    x = np.arange(len(x_values))

    w = 0.2
    ax.barh(x + (w / 2), y_values_confirmed, height=w, color='b', align='center', label='confirmed')
    ax.barh(x - (w / 2), y_values_deceased, height=w, color='g', align='center', label='deceased')

    x = list(x)

    # Add tally values to each bar
    for i, v in enumerate(y_values_confirmed):
        ax.text(v + .2, i + (w/2), str(v), color='b', fontsize=6)
    for i, v in enumerate(y_values_deceased):
        ax.text(v + .2, i - (w*2), str(v), color='g', fontsize=6)

    ax.axvline(0, color='k', lw=1)
    plt.yticks(x, x_values)
    plt.legend()
    ax.set_title('Top 20 Total cases by county')
    ax.grid(axis='x')
    plt.show()


