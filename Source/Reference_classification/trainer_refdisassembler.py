from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from data_processor import get_dataloaders_with_labels_and_relations
from sequence_classifier import RobertaCRF, RefDissassembler
from Utils.io_operations import load_jsonl_dataset
from sklearn.metrics import classification_report
from Source.Logging.loggers import get_logger
from Utils.labels import *
from utils import *
from tqdm import tqdm
import datetime
import torch
import time
import gc


CONFIG = {
    "OUTPUT_DIR": paths.trained_models_folder,
    "MODEL_NAME": 'FacebookAI/xlm-roberta-large',
    "DEVICE": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    "BATCH_SIZE": 8,
    "MAX_LENGTH": None,
    "NUM_CLASSES": 9,
    "NUM_RELATIONS": 1,
    "EPOCHS": 5,
    "LEARNING_RATE": 2e-5,
    "VERSION": "Disassembler_v1.0",
    "Comment": "Dataset with relation annotations. Updated labels.",
    "TRAINING_DATASET": "Annotated_dataset",
    "SPLIT_SIZE": 0.25,
    # "DATASET_VERSION": "VRT5.3",
    "DATASET_VERSION": "VD5.4_balanced",
    "VALIDATION_VERSION": "VDV5.4",
    "CHECKPOINT": [],
    "PIPELINE": ["TRAIN", "EVAL", "VALIDATE"]
    # "PIPELINE": ["TRAIN", "EVAL"]
}

logger = get_logger("RD_trainer_logger", "Training_logs.log")


def train(model, dataloader, device, epochs, learning_rate, val_set=None):
    # losses records
    losses = []
    # Training parameters
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    model.to(device)

    logger.info(f"\tStarting training")
    for epoch in range(epochs):
        logger.info(f"\t\tStarting epoch {epoch+1}")
        start = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Training Epoch {epoch}: "):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Removing the padding from the relations and flattening the labels
            relations = batch["relations"]
            relations = relations.tolist()
            processed_rel_batch = []
            for ex_relations in relations:
                if -1 in ex_relations:
                    processed_rel_batch += ex_relations[:ex_relations.index(-1)]
                else:
                    processed_rel_batch += ex_relations

            loss, rel_loss = model(input_ids, attention_mask, labels, processed_rel_batch)
            total_loss += loss.item()

            # if batch_count % 50 == 0:
            logger.info(f'\t\t\tTotal loss for batch : {loss.item()} The relation_extraction loss = {rel_loss.item()}')

            loss.backward()
            optimizer.step()
            scheduler.step()
            # optimizing by emptying cache and collecting garbage
            gc.collect()
            torch.cuda.empty_cache()

            if (batch_count == 0 or batch_count == len(dataloader)//2 or batch_count+1 == len(dataloader)) and \
                    val_set is not None:
                logger.info(f'\t\t\tValidation round ...')
                evaluate_model(model, val_set, device)
            batch_count += 1

        end = time.time()
        avg_train_loss = total_loss / len(dataloader)
        logger.info(
            f'Epoch {epoch+1}, Loss: {avg_train_loss} - runtime for epoch: {datetime.timedelta(seconds=end-start)}'
        )
        losses.append(avg_train_loss)

        if epoch + 1 in CONFIG["CHECKPOINT"] and epoch != epochs-1:
            logger.info(f"Saving trained model with config")
            save_model(CONFIG, model, losses, checkpoint=epoch)

    return losses


def evaluate_model(model, dataloader, device):
    # Transferring model to device
    model.to(device)
    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    predictions_rel, true_rel = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Evaluation "):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            relations = batch['relations'].to(device)
            relations = relations.tolist()
            processed_rel_batch = []
            for ex_relations in relations:
                if -1 in ex_relations:
                    processed_rel_batch += ex_relations[:ex_relations.index(-1)]
                else:
                    processed_rel_batch += ex_relations

            output_ref, output_rel = model(input_ids, attention_mask, labels)

            # Taking care of the labeling
            predictions.extend(output_ref)
            true_labels.extend([labels[i].tolist()[:len(o)] for i, o in enumerate(output_ref)])

            if output_rel is not None:
                predictions_rel.extend(output_rel)
                true_rel.extend(processed_rel_batch)

    # Convert predictions and labels to tag names
    pred_tags = [[ID2TAG[id] for id in pred] for pred in predictions]
    true_tags = [[ID2TAG[id] for id in true] for true in true_labels]

    # Flatten the lists for evaluation
    flat_pred_tags = [item for sublist in pred_tags for item in sublist]
    flat_true_tags = [item for sublist in true_tags for item in sublist]

    print("Label classification report")
    print(classification_report(flat_true_tags, flat_pred_tags, zero_division=0))
    print("\n\n================================\n\nRelation classification report")
    print(classification_report(true_rel, predictions_rel, zero_division=0))
    logger.info(f"\n {classification_report(flat_true_tags, flat_pred_tags)}")
    logger.info(f"\n {classification_report(true_rel, predictions_rel)}")


if __name__ == '__main__':
    # load training data from json file
    logger.info(f"Loading training data from json file: {paths.annotations_file}")
    dataset = load_jsonl_dataset(os.path.join(paths.annotations_folder,
                                              f"{CONFIG['TRAINING_DATASET']}_{CONFIG['DATASET_VERSION']}.jsonl"))

    # Loading validation set from the separate file
    logger.info(f"Loading training data from json file")
    validation_set = load_jsonl_dataset(os.path.join(
        paths.annotations_folder,
        f"{CONFIG['TRAINING_DATASET']}_{CONFIG['VALIDATION_VERSION']}.jsonl")
    )

    # Initialize the tokenizer
    logger.info(f"Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
    # initialize model
    logger.info(f"Initializing model")

    # if model version already exits load it for evaluation
    if os.path.exists(os.path.join(
            CONFIG["OUTPUT_DIR"],
            f"{CONFIG['MODEL_NAME'].split('/')[1]}_{str(CONFIG['VERSION'])}")):
        logger.info(f"Loading pretrained model Weights {CONFIG['MODEL_NAME']} version {CONFIG['VERSION']}")

        checkpoint = torch.load(os.path.join(CONFIG["OUTPUT_DIR"],
                                             f"{CONFIG['MODEL_NAME'].split('/')[1]}_{str(CONFIG['VERSION'])}",
                                             f"{CONFIG['MODEL_NAME'].split('/')[1]}_{str(CONFIG['VERSION'])}"))
        classifier = RefDissassembler(model_name=CONFIG['MODEL_NAME'], num_labels=checkpoint['num_labels'],
                                      num_relations=checkpoint['num_relations']
                                      )
        classifier.load_state_dict(checkpoint['model_state_dict'])

    else:
        classifier = RefDissassembler(CONFIG["MODEL_NAME"], CONFIG["NUM_CLASSES"], CONFIG["NUM_RELATIONS"])

    if "EVAL" in CONFIG["PIPELINE"]:
        # Split dataset into train and validation sets (for demonstration)
        dataset = dataset.train_test_split(test_size=CONFIG["SPLIT_SIZE"])
        # initialize dataloaders
        logger.info(f"Initializing dataloaders")
        train_dataloader = get_dataloaders_with_labels_and_relations(tokenizer,
                                                                     dataset["train"], CONFIG["BATCH_SIZE"],
                                                                     TAG2ID,
                                                                     CONFIG["MAX_LENGTH"]
                                                                     )

        eval_dataloader = (get_dataloaders_with_labels_and_relations(tokenizer,
                                                                     dataset["test"],
                                                                     CONFIG["BATCH_SIZE"],
                                                                     TAG2ID,
                                                                     CONFIG["MAX_LENGTH"]
                                                                     )
                           )
    else:
        # initialize dataloaders
        logger.info(f"Initializing dataloaders")
        train_dataloader = get_dataloaders_with_labels_and_relations(tokenizer,
                                                                     dataset, CONFIG["BATCH_SIZE"],
                                                                     TAG2ID,
                                                                     CONFIG["MAX_LENGTH"]
                                                                     )
        eval_dataloader = None

    validation_dataloader = None
    if "VALIDATE" in CONFIG["PIPELINE"]:
        logger.info(f"Initializing validation dataloader")
        validation_dataloader = get_dataloaders_with_labels_and_relations(tokenizer,
                                                                          validation_set,
                                                                          CONFIG["BATCH_SIZE"],
                                                                          TAG2ID,
                                                                          CONFIG["MAX_LENGTH"]
                                                                          )

    if "TRAIN" in CONFIG["PIPELINE"]:
        loss_record = train(classifier, train_dataloader, CONFIG["DEVICE"], CONFIG["EPOCHS"], CONFIG["LEARNING_RATE"],
                            validation_dataloader)
        logger.info(f"Saving trained model with config")
        # save_model(CONFIG, classifier, loss_record)

    if "EVAL" in CONFIG["PIPELINE"]:
        logger.info(f"Evaluating trained model")
        evaluate_model(classifier, eval_dataloader, CONFIG["DEVICE"])

