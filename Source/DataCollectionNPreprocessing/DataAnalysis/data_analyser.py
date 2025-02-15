from Source.Utils.labels import POSSIBLE_RELATIONS, TAG2ID
from Source.Utils.paths import *
from os import path
import json
from collections import Counter
import matplotlib.pyplot as plt


def count_entities_and_relations(file_path):
    entity_counter = Counter()
    relation_counter = Counter()
    counter = 0

    if path.exists(file_path):
        with open(file_path, "r", encoding='utf8') as file:
            for line in file:
                counter += 1
                data = json.loads(line)
                entities = data["entities"]
                relations = data["relations"]

                # Create a dictionary to map entity IDs to their types
                entity_id_to_type = {entity["id"]: entity["label"] for entity in entities}

                for entity in entities:
                    entity_counter[entity["label"]] += 1

                for relation in relations:
                    from_type = entity_id_to_type[relation["from_id"]]
                    to_type = entity_id_to_type[relation["to_id"]]
                    relation_counter[(from_type, to_type)] += 1

                    if TAG2ID[to_type] not in POSSIBLE_RELATIONS[TAG2ID[from_type]]:
                        emap = {}
                        for entity in data["entities"]:
                            emap[entity["id"]] = entity

                        print(counter)
                        print(data['text'])
                        print(len(data["relations"]))
                        for r in data["relations"]:
                            fe = emap[r["from_id"]]
                            te = emap[r["to_id"]]

                            print(f"{data['text'][fe['start_offset']:fe['end_offset']]} ({fe['id']})->"
                                  f"{data['text'][te['start_offset']:te['end_offset']]} ({te['id']})")
    return entity_counter, relation_counter


def plot_counter(counter, title, filename):
    plt.figure(figsize=(12, 6))
    labels = [f"{k[0]}->{k[1]}" if isinstance(k, tuple) else k for k in counter.keys()]
    plt.bar(labels, counter.values())
    plt.title(title)
    plt.xlabel('Types')
    plt.ylabel('Count')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


file_name = path.join(annotations_folder, "Annotated_dataset_VD5.5_Balanced.jsonl")
entity_counts, relation_counts = count_entities_and_relations(file_name)

plot_counter(entity_counts, 'Entity Type Occurrences', 'entity_occurrences.png')
plot_counter(relation_counts, 'Relation Type Occurrences', 'relation_occurrences.png')

print("Entity counts:", dict(entity_counts))
print("Relation counts:", {f"{k[0]}->{k[1]}": v for k, v in relation_counts.items()})
