from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
import os
import uuid

def save_report(targets, predicted, labels, skipped=None, append_file_name="", file_name=None):
    if not file_name:
        file_location=f"test_results/{uuid.uuid4()}/"
    else:
        file_location=f"test_results/{file_name}/"
        # TODO ask on cli if test wants to be replaced
    os.makedirs(file_location, exist_ok=True)

    report = classification_report(
        targets, predicted, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().round(2)
    if skipped is not None:
        report_df = pd.concat([report_df, skipped], axis=1)
    report_df.to_csv(f"{file_location}report{append_file_name}.csv")

# TODO fix these charts up
def apply_charts(cs_tickets_df, file_name=None):
    # getting target probabilities
    target_probs = cs_tickets_df['queue'].value_counts(normalize=True).round(2)

    size = 3
    fig = plt.figure(figsize=(10, 18))

    plt.subplot(size, 1, 1)
    cs_tickets_df.groupby("target")["confidence"].mean().plot(kind='bar')
    plt.title("Average Confidence by Target Category")
    plt.xlabel("Target")
    plt.ylabel("Average Confidence")

    plt.subplot(size, 1, 2)
    cs_tickets_df['queue'].value_counts().plot(kind='bar')
    plt.title("Request Count per Queue")
    plt.xlabel("Queue")
    plt.ylabel("Number of Requests")

    plt.subplot(size, 1, 3)
    cs_tickets_df['target'].value_counts().plot(kind='bar')
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
    if file_name:
        file_location=f"test_results/{file_name}/"
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(f"{file_location}support_analysis_charts.png")
        print("Charts saved as 'support_analysis_charts.png'")