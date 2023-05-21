import csv
import env_loader as e

# CONTROL SESSION
IS_CONTROL_SESSION = False
CONTROL_SESSION_TYPE = 'IZA-gpt-3.5-data-gpt-3.5'

control_question_filename = "control-session-questions/%s.csv" % CONTROL_SESSION_TYPE
fields = []
control_questions = []

filename = 'control-session-results/%s-%s.csv' % (e.openai_model, CONTROL_SESSION_TYPE)
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
