from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch

# Load the ELECTRA model for sequence classification
model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=6)
model.load_state_dict(torch.load('best_model_weights.pth', map_location=torch.device('cpu')))
model.eval()

# Load the ELECTRA tokenizer
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

def get_prediction(title: str, paragraph: str, timestamp: float) -> int:
    # Tokenize input text and convert to model input format
    encoding = tokenizer.encode_plus(
        title  + f'T[{timestamp}]',
        text_pair=paragraph,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Perform inference with the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    # Getting the label with max probability
    predicted_label = torch.argmax(logits, dim=1).item()

    # Get reverse labelling done for the classes
    reverse_label = {
      0: 'Environmental',
      1: 'Commodities',
      2: 'Delays',
      3: 'Financial Health',
      4: 'Compliance',
      5: 'Supplier Market'
    }
    return {
      "status":200,
      "data":{
        "predicted_category": reverse_label[predicted_label]
        }
      }
