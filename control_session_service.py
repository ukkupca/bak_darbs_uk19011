import csv
import time

import env_loader as e

# CONTROL SESSION
IS_CONTROL_SESSION = True
CONTROL_SESSION_TYPE = 'iza-gpt-3.5-turbo-origin'

control_question_filename = "control-session-questions/%s-%s.csv" % (CONTROL_SESSION_TYPE, 'control-questions')
fields = []
control_questions = []

filename = 'control-session-results/%s-%s-%s-%s.csv' % (CONTROL_SESSION_TYPE, e.openai_model, 'results', str(time.time()))
csv_fields = ['CQID', 'Model', 'AttemptNumber', 'Answer']
results = []


def import_control_questions():
    with open(control_question_filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        global fields
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            control_questions.append(row)


def add_result(result):
    results.append(result)


def save():
    # writing to csv file
    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(csv_fields)

        # writing the data rows
        csvwriter.writerows(results)
