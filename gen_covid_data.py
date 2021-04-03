"""
Author: John McDonald
This file generates random records for the COPIS database, and saves it to a file.
"""

import argparse
import numpy as np
import datetime
import random
import string
import csv

def generate_random_emails(nb, length):
    """
    Generates a list of random emails for several different domain names
    :param nb: Number of emails to generate
    :param length: Length for the email username
    :return: List of random emails generated
    """
    domains = ["hotmail.com", "gmail.com", "aol.com", "mail.com", "yahoo.com"]
    letters = string.ascii_lowercase[:8]
    return [(''.join(random.choice(letters) for i in range(length)) + '@' + random.choice(domains)) for i in range(nb)]

def generate_random_dates(no):
    """
    Generates random dates between March 1st, and the current date.
    :param nb: Number of dates to generate
    :return: List of random dates as datetime objects
    """

    start_date = datetime.date(2020, 3, 1)
    end_date = datetime.date.today()
    # time_between_dates = end_date - start_date
    # days_between_dates = time_between_dates.days
    dates = []

    for i in range(no):
        random_number_of_days = random.randrange((end_date - start_date).days)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)
        dates.append(random_date)

        # simulate an exponential increase
        start_date = start_date + datetime.timedelta(days=1)
        if start_date == end_date:
            start_date = datetime.date(2020, 3, 1)

    return dates

parser = argparse.ArgumentParser(description='Generate covid data for testing COPIS')
parser.add_argument("-no", default=100000, type=int, help="Number of random patients to be generated.")
parser.add_argument("-fname", default="coviddata.csv", type=str, help="File name for storage.")

# arguments
args = parser.parse_args()
no = args.no + 1
fname = args.fname

# Columns
Pno = list(range(1, no))

Age = np.random.choice(99, no)
Gen = np.random.choice(["M","F"], no, p=[.45,.55])

counties = ['Mecklenburg', 'Wake', 'Durham', 'Wayne', 'Guilford', 'Rowan', 'Chatham', 'Forsyth', 'Robeson', 'Cumberland', 'Cabarrus', 'Randolph', 'Union', 'Orange', 'Lee', 'Duplin', 'Henderson', 'Harnett', 'Wilson', 'Wilkes', 'Davidson', 'Johnston', 'Columbus', 'Gaston', 'Granville', 'Pitt', 'Vance', 'Alamance', 'Edgecombe', 'Rutherford', 'Iredell', 'Sampson', 'Burke', 'Hoke', 'Nash', 'Lenoir', 'Franklin', 'Moore', 'Northampton', 'New Hanover', 'Buncombe', 'Richmond', 'Halifax', 'Pasquotank', 'Catawba', 'Bertie', 'Caldwell', 'Onslow', 'Bladen', 'Craven', 'Brunswick', 'Cleveland', 'Yadkin', 'Hertford', 'Montgomery', 'Lincoln', 'Rockingham', 'Anson', 'Caswell', 'Pender', 'Scotland', 'Greene', 'Surry', 'Davie', 'Person', 'Carteret', 'Martin', 'McDowell', 'Polk', 'Stanly', 'Washington', 'Beaufort', 'Warren', 'Jackson', 'Cherokee', 'Jones', 'Perquimans', 'Haywood', 'Dare', 'Gates', 'Stokes', 'Alexander', 'Chowan', 'Currituck', 'Watauga', 'Ashe', 'Pamlico', 'Alleghany', 'Transylvania', 'Yancey', 'Clay', 'Mitchell', 'Swain', 'Tyrrell', 'Macon', 'Camden', 'Graham', 'Hyde', 'Madison', 'Avery']
p = [0.142290013, 0.070553002, 0.058991503, 0.050564145, 0.039977713, 0.03343084, 0.03113247, 0.025699958, 0.025421368, 0.024167711, 0.023749826, 0.022078284, 0.020476389, 0.017272601, 0.017133305, 0.015670706, 0.015601059, 0.014765288, 0.014138459, 0.014068812, 0.013999164, 0.013999164, 0.013650926, 0.011700794, 0.011422204, 0.011282908, 0.011213261, 0.010586433, 0.010516785, 0.010516785, 0.010238195, 0.0100989, 0.009123833, 0.00835771, 0.008218415, 0.007870177, 0.007730882, 0.007661234, 0.007104053, 0.006686168, 0.006407578, 0.006198635, 0.005920045, 0.005641454, 0.005432511, 0.004178855, 0.003830617, 0.003760969, 0.003552027, 0.003552027, 0.003482379, 0.003482379, 0.003412732, 0.003343084, 0.002855551, 0.002716256, 0.002716256, 0.002646608, 0.002646608, 0.002507313, 0.002507313, 0.00229837, 0.00229837, 0.002228723, 0.002089427, 0.00201978, 0.00201978, 0.00201978, 0.00201978, 0.00201978, 0.00174119, 0.001671542, 0.001601894, 0.001462599, 0.001253656, 0.001253656, 0.001184009, 0.001114361, 0.001044714, 0.000766123, 0.000766123, 0.000696476, 0.000696476, 0.000626828, 0.000626828, 0.000557181, 0.000557181, 0.000487533, 0.000487533, 0.000487533, 0.000348238, 0.000348238, 0.000348238, 0.00027859, 0.000208943, 0.000139295, 0.000139295, 0.0, 0.0, 0.0]
County = np.random.choice(counties, no, p)

IStatus = np.random.choice(['U','N','P'], no, [0.2,0.7,0.1])
PStatus = np.random.choice(['U','Q','H','I','R','D'], no, [.25,.15,.06,.02,.5,.02])

Email = generate_random_emails(no,8)

# Dates and history
dates = generate_random_dates(no)
Tdt = []
History = []

for date, pstatus, istatus in zip(dates, PStatus, IStatus):
    i_date = date - datetime.timedelta(days = 5)
    Tdt.append(date.strftime("%B, %d, 20%y"))
    History.append(istatus + ":" + i_date.strftime("%B, %d, 20%y") + ", " + pstatus + ':' + date.strftime("%B, %d, 20%y"))

Notes = list(['']*no)

zipped = list(zip(Pno,Age,Gen,County,IStatus,PStatus,Tdt,Email,History,Notes))

with open(fname, 'w', newline="") as new_file:

    fieldnames = ['Pno','Age','Gen','County','IStatus','PStatus',
                  'Tdt','Email','History','Notes']
    csv_writer = csv.writer(new_file, delimiter='\t')
    csv_writer.writerow(fieldnames)
    csv_writer.writerows(zipped)

print('Generated {0} records to {1}'.format(no-1, fname))