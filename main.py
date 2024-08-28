import pandas as pd

student_data = pd.read_csv('student-mat.csv', sep=';')

X = student_data.drop(columns = ['G1', 'G2', 'G3'])
y = student_data[['G1', 'G2', 'G3']]

  
 
  
# data (as pandas dataframes) 
# X = student_performance.data.features 
# y = student_performance.data.targets 
  
# # metadata 
# print(student_performance.metadata) 
  
# # variable information 
# print(student_performance.variables) 
