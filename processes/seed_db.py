import pandas as pd
import matplotlib.pyplot as plt

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/Tobi-Bueck/customer-support-tickets/dataset-tickets-multi-lang-4-20k.csv")


# dropping german rows
german = df['language'] == 'de'

df = df[~german]

# removing unwanted cols
columns = df.columns.tolist()
columns_keep = ['subject', 'body', 'priority', 'queue']

drop_columns = [c for c in columns if c not in columns_keep]

df.drop(columns=drop_columns, inplace=True)



# dropping rows that has an empty 
df = df.dropna(how='any')



TECHNICAL = 'technical'
BILLING = 'billing'
GENERAL = 'general'

# map queue
queue_map = {
    'Technical Support': TECHNICAL, 
    'IT Support': TECHNICAL,
    'Billing and Payments': BILLING,
    'Customer Service': GENERAL,
    'Product Support': GENERAL,
    'Service Outages and Maintenance': GENERAL,
    'Human Resources': GENERAL,
    'Sales and Pre-Sales': GENERAL,
    'Returns and Exchanges': GENERAL,
    'General Inquiry': GENERAL
}
df['target'] = df['queue'].map(queue_map)

if df['target'].isna().any():
    raise AssertionError('No Nan allowed in "target"')

# getting target probabilities
target_probs = df['target'].value_counts(normalize=True).round(2)

# map parse priority to float
confidence_map = {
    'low': 0.1,
    'medium': 0.5,
    'high': 0.95
}
df['confidence'] = df['priority'].map(confidence_map)

if df['confidence'].isna().any():
    raise AssertionError('No Nan allowed in "confidence"')


size = 3
fig = plt.figure(figsize=(10, 18))

plt.subplot(size, 1, 1)
df.groupby("target")["confidence"].mean().plot(kind='bar')
plt.title("Average Confidence by Target Category")
plt.xlabel("Target")
plt.ylabel("Average Confidence")

plt.subplot(size, 1, 2)
df['queue'].value_counts().plot(kind='bar')
plt.title("Request Count per Queue")
plt.xlabel("Queue")
plt.ylabel("Number of Requests")

plt.subplot(size, 1, 3)
df['target'].value_counts().plot(kind='bar')
plt.title("Request Count per Target")
plt.xlabel("Queue")
plt.ylabel("Number of Requests")

# Adding other information as text
prob_text = "\n".join([f"• {target.capitalize()}: {prob * 100:.0f}% of requests"
                       for target, prob in target_probs.items()])
summary_text = (
    "Target Probabilities:\n"
    f"{prob_text}\n\n"
    "These probabilities reflect how likely a given request is to fall into each category "
    "based on observed data."
)
fig.text(0.05, 0.02, summary_text, ha='left', va='bottom', fontsize=10)


# Saving
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("support_analysis_charts.png")
print("Charts saved as 'support_analysis_charts.png'")


# Make charts showing how balanced the database is
# Add Base probities

df.to_csv("customer_support_tickers_dataset.csv")

print(df)