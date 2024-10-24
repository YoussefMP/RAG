from Source.Utils.io_operations import load_jsonl_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from Source.Utils import paths
import os


def bar_plot(x, y):
    # Create a bar plot for length_data
    plt.figure(figsize=(10, 6))

    # Plot the bar graph
    plt.bar(x, y)

    # Set the x-axis and y-axis labels
    plt.xlabel('Length of Text')
    plt.ylabel('Number of Occurrences')

    # Set the title of the plot
    plt.title('Distribution of Text Lengths')

    # Show the plot
    plt.show()


def report_length(tokenizer, data):

    long_refs = []
    report_result = {}
    tokenized_lengths = []
    text_length = []
    for entry in data:
        tokenized_input = tokenizer.tokenize(entry["text"])

        if len(tokenized_input) > 96:
            text_length.append(len(entry["text"]))
            tokenized_lengths.append(len(tokenized_input))
            if len(tokenized_input) in report_result:
                report_result[len(tokenized_input)] += 1
            else:
                report_result[len(tokenized_input)] = 1
            long_refs.append(entry["id"])

            print(entry["text"])

    bar_plot(list(report_result.keys()), list(report_result.values()))


if __name__ == '__main__':
    model_name = "FacebookAI/xlm-roberta-large"
    roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_name = "Annotated_dataset_VD5.3_balanced.jsonl"
    dataset = load_jsonl_dataset(os.path.join(paths.annotations_folder, dataset_name))

    report_length(roberta_tokenizer, dataset)
