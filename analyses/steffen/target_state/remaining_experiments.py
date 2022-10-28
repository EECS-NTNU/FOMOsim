import csv

finished_tasks = []
with open('completed_tasks.csv', newline='') as f:
    for row in csv.reader(f):
        finished_tasks.append(int(row[0]))

print(len(finished_tasks))

max_task =60
remaining_tasks = list(set(list(range(max_task)))-set(finished_tasks))
print(len(remaining_tasks))
print(remaining_tasks)


