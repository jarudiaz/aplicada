import pandas as pd
pd.options.display.max_colwidth = 100
test_data = pd.read_csv('test_data.csv', usecols=['sentence', 'sentiment'])

print(test_data)