import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import os
import docx2txt

# Directory where your course files are stored
course_directory = "data\courses\zqm_modul\en"

# Initialize an empty list to store course descriptions
course_descriptions = {}

# Loop through each file in the directory
for filename in os.listdir(course_directory):
    if filename.endswith(".docx"):
        # Read the contents of the Word file
        file_path = os.path.join(course_directory, filename)
        course_content = docx2txt.process(file_path)
        
        # Append course content to the list
        course_descriptions[filename] = course_content

# Now you have course descriptions in the `course_descriptions` dictionary

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
model = AutoModel.from_pretrained("thenlper/gte-large")

# Tokenize and process course descriptions
course_embeddings = []
course_descriptions_list = []

i = 0 
 
for filename, content in course_descriptions.items():
    # Tokenize the content of each document
    if i<22:
        i+=1
        continue
    tokenized_content = tokenizer(content, max_length=512, padding=True, truncation=True, return_tensors='pt')
    
    # Pass tokenized content through the model
    outputs = model(**tokenized_content)
    embeddings = outputs.last_hidden_state
    
    # Average pooling to get a single embedding for each course description
    attention_mask = tokenized_content['attention_mask']
    pooled_embedding = embeddings.mean(dim=1)
    
    # Normalize the embeddings
    normalized_embedding = F.normalize(pooled_embedding, p=2, dim=1)
    
    # Append the normalized embedding to the list
    course_embeddings.append(normalized_embedding)
    
    # Store the filename for later reference
    course_descriptions_list.append(filename)
    i+=1
    print(course_descriptions_list)


# course_embeddings_tensor = torch.cat(course_embeddings, dim=0)
# torch.save(course_embeddings_tensor, 'temp files/Embedding/en/embeddingafter22.pt')

print("next 19 all files got Saved")