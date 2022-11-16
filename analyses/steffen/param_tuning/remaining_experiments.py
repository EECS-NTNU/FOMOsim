import csv

finished_tasks = []
with open('completed_tasks.csv', newline='') as f:
    for row in csv.reader(f):
        finished_tasks.append(int(row[0]))

print(len(finished_tasks))

finished_tasks2 = []
with open('completed_tasks2.csv', newline='') as f:
    for row in csv.reader(f):
        finished_tasks2.append(int(row[0]))

print(len(finished_tasks2))


max_task =2415
remaining_tasks = list(set(list(range(max_task)))-set(finished_tasks)-set(finished_tasks2))
print(len(remaining_tasks))

2288+82