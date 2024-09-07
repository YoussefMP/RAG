from Utils.io_operations import load_jsonl_dataset
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from sequence_classifier import RobertaCRF, RefDissassembler
from sklearn.metrics import classification_report
from Source.Logging.loggers import get_logger
from Utils.labels import *
from utils import *
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
    "NUM_CLASSES": 9,
    "NUM_RELATIONS": 1,
    "EPOCHS": 5,
    "LEARNING_RATE": 2e-5,
    "VERSION": "Disassembler_v1.0",
    "Comment": "Dataset with relation annotations. Updated labels.",
    "TRAINING_DATASET": "Annotated_dataset",
    "EVAL": False,
    "SPLIT_SIZE": 0.4,
    "DATASET_VERSION": "VRT5.2",
    "CHECKPOINT": [],
}

TRAIN = True

logger = get_logger("trainer_logger", "Training_logs.log")


def save_model(model, loss_records, checkpoint=None):
    logger.info(f"Saving trained model with config")
    out_path = os.path.join(CONFIG["OUTPUT_DIR"],
                            f"{CONFIG['MODEL_NAME'].split('/')[1]}_{CONFIG['VERSION']}"
                            )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model_name = f"{CONFIG['MODEL_NAME'].split('/')[1]}_{CONFIG['VERSION']}"

    torch.save({
        'model_state_dict': model.state_dict(),
        'num_labels': CONFIG["NUM_CLASSES"],
        'model_name': model_name + f"_checkpoint{checkpoint}" if checkpoint else model_name
    }, os.path.join(out_path, model_name + f"_checkpoint{checkpoint}"))

    CONFIG["Losses"] = loss_records
    save_config(out_path, **CONFIG)


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

            # Removing the padding from the relations and flattening the labels
            relations = batch["relations"]
            relations = relations.tolist()
            processed_rel_batch = []
            for ex_relations in relations:
                if -1 in ex_relations:
                    processed_rel_batch += ex_relations[:ex_relations.index(-1)]
                else:
                    processed_rel_batch += ex_relations

            loss = model(input_ids, attention_mask, labels, processed_rel_batch)
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

        if epoch + 1 in int(CONFIG["CHECKPOINT"]) and epoch != epochs-1:
            save_model(model, losses, checkpoint=epoch)

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

            output_ref, output_rel = model(input_ids, attention_mask)

            # Taking care of the labeling
            predictions.extend(output_ref)
            true_labels.extend([labels[i].tolist()[:len(o)] for i, o in enumerate(output_ref)])

            predictions.extend(output_rel)
            true_rel.extend([relations[i].tolist()[:len(o)]] for i, o in enumerate(output_rel))

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
    dataset = load_jsonl_dataset(os.path.join(paths.annotations_folder,
                                              f"{CONFIG['TRAINING_DATASET']}_{CONFIG['DATASET_VERSION']}.jsonl"))
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

        TRAIN = False
    else:
        classifier = RefDissassembler(CONFIG["MODEL_NAME"], CONFIG["NUM_CLASSES"], CONFIG["NUM_RELATIONS"])

    if CONFIG["EVAL"]:
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

    if TRAIN:
        loss_record = train(classifier, train_dataloader, CONFIG["DEVICE"], CONFIG["EPOCHS"], CONFIG["LEARNING_RATE"])
        save_model(classifier, loss_record)

    if CONFIG["EVAL"]:
        logger.info(f"Evaluating trained model")
        evaluate_model(classifier, eval_dataloader, CONFIG["DEVICE"])

