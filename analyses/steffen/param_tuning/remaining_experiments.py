import csv

finished_tasks = []
with open('completed_tasks.csv', newline='') as f:
    for row in csv.reader(f):
        finished_tasks.append(int(row[0]))


max_task =2415
remaining_tasks = list(set(list(range(max_task)))-set(finished_tasks))
