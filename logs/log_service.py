import csv
import time
import env_loader as e

# LOGS
SAVE_LOGS = True
csv_fields = ['Role', 'Type', 'Content']
logs = []
filename = 'logs/%s-%s.csv' % (e.log_name, int(time.time()))


def add(row):
    logs.append(row)


def save():
    # writing to csv file
    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(csv_fields)

        # writing the data rows
        csvwriter.writerows(logs)


FILE_NAME = 'gpt-3-origin-load'
load_fields = []
load_rows = []
file_path = 'C:/Users/uk15006/PycharmProjects/CHAT_LANG/gpt-3-origin-load.csv'


def load():
    with open(file_path, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        global load_fields
        load_fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            load_rows.append(row)

        return load_rows
