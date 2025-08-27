import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.ticker import FuncFormatter

# Set a professional style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 1. Load the dataset
df = pd.read_excel('OnlineRetail.xlsx')
print ('==== LOAD DATA ===')
print ('First 5 rows : ')
print(df.head(5))

# 2. Data cleaning 
print("=== MISSING VALUES ===")
print(df.isnull().sum())
# Remove rows where CustomerID is missing
df.dropna(subset=['CustomerID'],inplace=True)
print('=== MISSING VALUES AFTER TRAINING ===')
print(df.isnull().sum())
print(df.info())

## Convert CustomerID to string
df['CustomerID'] = df['CustomerID'].astype(str)
# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# Create a copy of the original dataframe before dropping columns
df_original = df.copy()
# Delete the column InvoiceNo
df.drop('InvoiceNo', axis='columns',inplace=True)
print('\n==== DATA TYPES AFTER CONVERSION ===')
print(df.head(5))
print(df.info())


# 3. Basic statistics exploration
# Create TotalPrice and time-based columns
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
# Create a function to format large numbers
def format_large_numbers(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.1f}K'
    else:
        return f'{x:.0f}'

formatter = FuncFormatter(format_large_numbers)

# 4. Enhanced visualizations
# Set a consistent color palette
colors = sns.color_palette("husl", 10)
# Figure 1: Top 10 countries by sales
country_sales = df.groupby('Country')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(14, 8))
ax = sns.barplot(x=country_sales.values, y=country_sales.index, palette=colors)
plt.title('Top 10 Countries by Quantity Sold', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Total Quantity Sold', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.tight_layout()
plt.show()

# Figure 2: Top 10 best-selling products
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(14, 10))
ax = sns.barplot(x=top_products.values, y=top_products.index, palette=colors)
plt.title('Top 10 Best-Selling Products', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Total Quantity Sold', fontsize=12)
plt.ylabel('Product Description', fontsize=12)
plt.tight_layout()
plt.show()

# Figure 3: Monthly sales trend
monthly_sales = df.groupby('InvoiceMonth')['TotalPrice'].sum()

plt.figure(figsize=(14, 7))
ax = monthly_sales.plot(kind='line', marker='o', linewidth=2.5, markersize=8, color=colors[0])
plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 4: Daily sales trend
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_sales = df.groupby('DayOfWeek')['TotalPrice'].sum().reindex(days_order)

plt.figure(figsize=(12, 7))
ax = sns.barplot(x=daily_sales.index, y=daily_sales.values, palette="viridis")
plt.title('Sales Trend by Day of the Week', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Day of the Week', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.tight_layout()
plt.show()

# Figure 6: UnitPrice distribution with outliers
plt.figure(figsize=(14, 7))

# Create subplots
plt.subplot(1, 2, 1)
sns.boxplot(y=df['UnitPrice'], color=colors[2])
plt.title('Unit Price Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Unit Price (£)')

plt.subplot(1, 2, 2)
# Filter out extreme outliers for better visualization
filtered_prices = df[df['UnitPrice'] < df['UnitPrice'].quantile(0.99)]['UnitPrice']
sns.histplot(filtered_prices, kde=True, color=colors[4])
plt.title('Unit Price Distribution (99th percentile)', fontsize=14, fontweight='bold')
plt.xlabel('Unit Price (£)')

plt.tight_layout()
plt.show()

# Print descriptive statistics
print('\n==== DESCRIPTIVE STATISTICS FOR UNITPRICE ===')
print(df['UnitPrice'].describe())

# Correlation heatmap (for numerical columns)
numerical_df = df[['Quantity', 'UnitPrice', 'TotalPrice']]
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()