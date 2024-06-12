from Utils.io_operations import load_jsonl_dataset
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import classification_report
from Source.Logging.loggers import get_logger
from sequence_classifier import RobertaCRF
from Utils.labels import *
from sequence_labeler.utils import *
from data_processor import *
from tqdm import tqdm
import datetime
import time
import gc


CONFIG = {
    "OUTPUT_DIR": paths.pretrained_classifiers_folder,
    "MODEL_NAME": 'FacebookAI/xlm-roberta-large',
    "DEVICE": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    # "DEVICE": "cpu",
    "BATCH_SIZE": 8,
    "MAX_LENGTH": 256,
    "NUM_CLASSES": 2,
    "EPOCHS": 5,
    "LEARNING_RATE": 2e-5,
    "VERSION": "r0.5",
    "Comment": "New annotation with longer sequence, more context",
    "TRAINING_DATASET": "annotated_dataset_long.jsonl"
}
TRAIN = True


logger = get_logger("trainer_logger", "Training_logs.log")


def train(model, dataloader, device, epochs, learning_rate):
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
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Training Epoch {epoch}: "):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            logger.info(f'\t\t\tTotal loss for batch : {total_loss}')
            loss.backward()
            optimizer.step()
            scheduler.step()

            # optimizing by emptying cache and collecting garbage
            gc.collect()
            torch.cuda.empty_cache()

        end = time.time()
        avg_train_loss = total_loss / len(dataloader)
        logger.info(
            f'Epoch {epoch+1}, Loss: {avg_train_loss} - runtime for epoch: {datetime.timedelta(seconds=end-start)}'
        )
        losses.append(avg_train_loss)

    return losses


def evaluate_model(model, dataloader, device):
    # Transferring model to device
    model.to(device)
    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Evaluation "):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, attention_mask)
            predictions.extend(output)
            true_labels.extend([labels[i].tolist()[:len(o)] for i, o in enumerate(output)])

    # Convert predictions and labels to tag names
    pred_tags = [[ID2TAG[id] for id in pred] for pred in predictions]
    true_tags = [[ID2TAG[id] for id in true] for true in true_labels]

    # Flatten the lists for evaluation
    flat_pred_tags = [item for sublist in pred_tags for item in sublist]
    flat_true_tags = [item for sublist in true_tags for item in sublist]

    print(classification_report(flat_true_tags, flat_pred_tags))
    logger.info(classification_report(flat_true_tags, flat_pred_tags))


if __name__ == '__main__':
    # load training data from json file
    logger.info(f"Loading training data from json file: {paths.annotations_file}")
    dataset = load_jsonl_dataset(os.path.join(paths.annotations_folder, CONFIG["TRAINING_DATASET"]))

    # Initialize the tokenizer
    logger.info(f"Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
    # initialize model
    logger.info(f"Initializing model")

    # if model version already exits load it for evaluation
    if os.path.exists(os.path.join(
            CONFIG["OUTPUT_DIR"],
            f"{CONFIG['MODEL_NAME'].split('/')[1]}_v{str(CONFIG['VERSION'])}")):
        logger.info(f"Loading pretrained model Weights {CONFIG['MODEL_NAME']} version {CONFIG['VERSION']}")
        classifier = torch.load(
            os.path.join(CONFIG["OUTPUT_DIR"], f"{CONFIG['MODEL_NAME'].split('/')[1]}_v{str(CONFIG['VERSION'])}")
        )
        TRAIN = False
    else:
        classifier = RobertaCRF(CONFIG["MODEL_NAME"], CONFIG["NUM_CLASSES"])

    # Split dataset into train and validation sets (for demonstration)
    dataset = dataset.train_test_split(test_size=0.2)
    # initialize dataloaders
    logger.info(f"Initializing dataloaders")
    train_dataloader = get_dataloaders_with_labels(tokenizer,
                                                   dataset["train"], CONFIG["BATCH_SIZE"],
                                                   TAG2ID,
                                                   CONFIG["MAX_LENGTH"]
                                                   )
    eval_dataloader = get_dataloaders_with_labels(tokenizer,
                                                  dataset["test"],
                                                  CONFIG["BATCH_SIZE"],
                                                  TAG2ID,
                                                  CONFIG["MAX_LENGTH"]
                                                  )

    if TRAIN:
        losses = train(classifier, train_dataloader, CONFIG["DEVICE"], CONFIG["EPOCHS"], CONFIG["LEARNING_RATE"])
        logger.info(f"Saving trained model with config")
        torch.save(classifier, os.path.join(
            CONFIG["OUTPUT_DIR"],
            f"{CONFIG['MODEL_NAME'].split('/')[1]}_{CONFIG['VERSION']}")
                   )
        CONFIG["Losses"] = losses
        save_config(**CONFIG)

    logger.info(f"Evaluating trained model")
    evaluate_model(classifier, eval_dataloader, CONFIG["DEVICE"])
