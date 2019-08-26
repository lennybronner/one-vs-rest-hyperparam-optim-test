from collections import defaultdict
import csv

def read_data(data_path):
    with open(data_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            yield {
                'content': row['content'], 
                'source_type': row['source_type'], 
                'parent_tag': row['parent_tag'], 
                'child_tag': row['child_tag']
            }

def generate_text_to_labels(data, label_type):
    source_type_to_text_to_labels = defaultdict(lambda: defaultdict(set))
    twitter_max_length = 280
    fb_insta_max_lengths = 1500
    for row in data:
        text = row['content']
        source_type = row['source_type'].lower()
        if source_type == "twitter":
            if len(text) < twitter_max_length:
                source_type_to_text_to_labels[source_type][text].add(row[label_type])
            else:
                source_type_to_text_to_labels['video'][text].add(row[label_type])
        else:
            if len(text) < fb_insta_max_lengths:
                source_type_to_text_to_labels[source_type][text].add(row[label_type])
            else:
                source_type_to_text_to_labels['video'][text].add(row[label_type])
    return source_type_to_text_to_labels

def extract_text_and_labels(text_to_labels):
    text = list(text_to_labels.keys())
    labels = [tuple(y) for x, y in text_to_labels.items()]
    return text, labels

def get_all_data(label_type='child_tag'):
    data_path = "all-annotations.txt"
    data = read_data(data_path)
    source_type_to_text_to_labels = generate_text_to_labels(data, label_type)
    twitter_texts, twitter_labels = extract_text_and_labels(source_type_to_text_to_labels['twitter'])
    fb_texts, fb_labels = extract_text_and_labels(source_type_to_text_to_labels['facebook'])
    insta_texts, insta_labels = extract_text_and_labels(source_type_to_text_to_labels['instagram'])
    video_texts, video_labels = extract_text_and_labels(source_type_to_text_to_labels['video'])
    return {
        'twitter_texts': twitter_texts,
        'twitter_labels': twitter_labels,
        'facebook_texts': fb_texts,
        'facebook_labels': fb_labels,
        'instagram_texts': insta_texts,
        'instagram_labels': insta_labels,
        'video_texts': video_texts,
        'video_labels': video_labels
        }

def get_texts_and_labels(from_):
    all_data = get_all_data()
    return all_data[f'{from_}_texts'], all_data[f'{from_}_labels']
