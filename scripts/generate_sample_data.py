import os
import pandas as pd
import numpy as np

os.makedirs('sample_data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Create dirty_customer_dataset
dirty_data = {
    'Customer_ID': [f'CUST_{1000+i}' for i in range(25)],
    'Name': [
        'Aarav Sharma', 'Diya Patel', 'Vivaan Gupta', 'Ananya Iyer', 'Vivaan Gupta',
        'Rohan Verma', 'Pooja Reddy', 'Rahul Nair', 'Sneha Kulkarni', 'Vikram Singh',
        'Ishita Rao', 'Arjun Mehta', 'Kavya Joshi', 'Aditya Mishra', 'Siddharth Jain',
        'Meera Pillai', 'Karan Malhotra', 'Tanvi Saxena', 'Rishi Choudhury', 'Nisha Sen',
        'Aman Bhatia', 'Rhea Kapoor', 'Mohit Aggarwal', 'Deepika Das', 'Sanjay Kumar'
    ],
    'Age': [
        '28', '34', '-5', '29', '-5',
        '42', '999', 'Thirty', '31', '45',
        '24', '38', '52', np.nan, '27',
        '33', '41', '26', '60', '35',
        '25', '29', '32', '47', '39'
    ],
    'City': [
        'Mumbai', 'mumbai', 'MUMBAI', ' Mumbai', 'MUMBAI',
        'Delhi', 'delhi', 'DELHI', 'Mumabi', 'Delhi',
        'Bangalore', 'Bangalore', 'Banglore', 'Bangalore', 'Pune',
        'Mumbai', 'Mumbai', 'Delhi', 'Pune', 'pune',
        'Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Mumbai'
    ],
    'Join_Date': [
        '2024-01-15', '2023-11-20', '22/13/2020', '2024-05-10', '22/13/2020',
        '2025-02-30', '2023-08-19', '2024-03-01', '2022-12-14', '2024-02-28',
        '2023-06-25', '2024-07-12', '2022-09-30', '2024-04-18', '2023-10-05',
        '2024-08-22', '2023-03-15', '2024-09-01', '2022-11-11', '2023-05-19',
        '2024-10-10', '2023-01-20', '2024-06-08', '2022-07-04', '2024-11-30'
    ],
    'Salary': [
        65000, 82000, 55000, 71000, 55000,
        95000, 1200000, 68000, 74000, 89000,
        62000, 85000, 110000, 58000, 76000,
        91000, 84000, 67000, 105000, np.nan,
        73000, 79000, 88000, 69000, 94000
    ],
    'Department': [
        'Engineering', 'Marketing', 'Engineering', 'Finance', 'Engineering',
        'Engineering', 'Sales', 'Finance', 'Marketing', 'Sales',
        'Engineering', 'Finance', 'Marketing', 'Sales', 'Engineering',
        'Marketing', 'Finance', 'Sales', 'Engineering', 'Finance',
        'Sales', 'Engineering', 'Marketing', 'Finance', 'Sales'
    ],
    'Churn': [
        0, 0, 1, 0, 1,
        0, 1, 0, 0, 1,
        0, 0, 0, 1, 0,
        0, 1, 0, 0, 1,
        0, 0, 0, 1, 0
    ]
}

df_dirty = pd.DataFrame(dirty_data)
# Make row 4 an exact duplicate of row 2
df_dirty.iloc[4] = df_dirty.iloc[2]

df_dirty.to_csv('sample_data/dirty_customer_dataset.csv', index=False)
df_dirty.to_excel('sample_data/dirty_customer_dataset.xlsx', index=False)
print('Created dirty_customer_dataset.csv and .xlsx')

# 2. Create clean_benchmark_dataset for corruption & ML evaluation
np.random.seed(42)
n_clean = 150
clean_data = {
    'Customer_ID': [f'CUST_{2000+i}' for i in range(n_clean)],
    'Age': np.random.randint(22, 65, size=n_clean),
    'Salary': np.random.normal(75000, 15000, size=n_clean).round(2),
    'Credit_Score': np.random.randint(580, 850, size=n_clean),
    'Tenure_Months': np.random.randint(1, 72, size=n_clean),
    'Department': np.random.choice(['Engineering', 'Marketing', 'Finance', 'Sales', 'Operations'], size=n_clean),
    'City': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Hyderabad', 'Chennai'], size=n_clean),
    'Join_Date': pd.date_range('2021-01-01', periods=n_clean, freq='W').strftime('%Y-%m-%d'),
    'Churn': np.random.choice([0, 1], p=[0.75, 0.25], size=n_clean)
}
df_clean = pd.DataFrame(clean_data)
df_clean.to_csv('sample_data/clean_benchmark_dataset.csv', index=False)
print('Created clean_benchmark_dataset.csv')
