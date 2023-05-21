import csv
import time
import env_loader as e

# LOGS
SAVE_LOGS = False
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
