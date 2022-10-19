from pandas import *

xls = ExcelFile('completed_tasks.xlsx')
df = xls.parse(xls.sheet_names[0])
print(df.to_dict())