# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel
# import os
# import docx2txt

# # Directory where your course files are stored
# # remove en with de for de files !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# course_directory = "temp files\jobs_txt"

# # Initialize an empty list to store course descriptions
# course_descriptions = {}
# course_descriptions_list = []

# for filename in os.listdir(course_directory):
#     if filename.endswith(".txt"):
#         # Read the contents of the text file
#         file_path = os.path.join(course_directory, filename)
#         with open(file_path, "r", encoding="utf-8") as file:
#             course_content = file.read()
        
#         # Add the course content to the dictionary with the filename as the key
#         course_descriptions[filename] = course_content
#         course_descriptions_list.append(filename)

# # Now you have course descriptions in the `course_descriptions` dictionary
# # Initialize tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
# model = AutoModel.from_pretrained("thenlper/gte-large")
# course_embeddings = []
# i = 1
# for filename, content in course_descriptions.items():
#     print(i)
#     tokenized_content = tokenizer(content, max_length=512, padding=True, truncation=True, return_tensors='pt')
#     # Pass tokenized content through the model
#     outputs = model(**tokenized_content)
#     embeddings = outputs.last_hidden_state
    
#     # Average pooling to get a single embedding for each course description
#     attention_mask = tokenized_content['attention_mask']
#     pooled_embedding = embeddings.mean(dim=1)
    
#     # Normalize the embeddings
#     normalized_embedding = F.normalize(pooled_embedding, p=2, dim=1)
    
#     # Append the normalized embedding to the list
#     course_embeddings.append(normalized_embedding)

#     course_embeddings_tensor = torch.cat(course_embeddings, dim=0)
#     if i % 25 == 0:

#         torch.save(course_embeddings_tensor, f'temp files\Embedding2\_{i}.pt')
#         course_embeddings = []
#     i+=1


# print("ddddddddddddddddddddddddddd")








# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel
# import os
# import docx2txt

# # Directory where your course files are stored
# # remove en with de for de files !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# course_directory = "temp files\jobs_txt"

# # Initialize an empty list to store course descriptions
# course_descriptions = {}
# course_descriptions_list = []

# for filename in os.listdir(course_directory):
#     if filename.endswith(".txt"):
#         # Read the contents of the text file
#         file_path = os.path.join(course_directory, filename)
#         with open(file_path, "r", encoding="utf-8") as file:
#             course_content = file.read()
        
#         # Add the course content to the dictionary with the filename as the key
#         course_descriptions[filename] = course_content
#         course_descriptions_list.append(filename)

# # Now you have course descriptions in the `course_descriptions` dictionary
# # Initialize tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
# model = AutoModel.from_pretrained("thenlper/gte-large")

# # Define batch size
# batch_size = 25

# # Iterate over course descriptions in batches
# for i in range(0, len(course_descriptions_list), batch_size):
#     batch_files = course_descriptions_list[i:i+batch_size]
#     batch_embeddings = []
    
#     for filename in batch_files:
#         content = course_descriptions[filename]
#         tokenized_content = tokenizer(content, max_length=512, padding=True, truncation=True, return_tensors='pt')
#         # Pass tokenized content through the model
#         outputs = model(**tokenized_content)
#         embeddings = outputs.last_hidden_state

#         # Average pooling to get a single embedding for each course description
#         attention_mask = tokenized_content['attention_mask']
#         pooled_embedding = embeddings.mean(dim=1)

#         # Normalize the embeddings
#         normalized_embedding = F.normalize(pooled_embedding, p=2, dim=1)

#         # Append the normalized embedding to the list
#         batch_embeddings.append(normalized_embedding)

#     # Concatenate embeddings in the batch
#     batch_embeddings_tensor = torch.cat(batch_embeddings, dim=0)

#     # Save the batch embeddings to a .pt file
#     torch.save(batch_embeddings_tensor, f'temp files\Embedding2\_{i}.pt')

# print("Embeddings saved successfully.")





# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel
# import os

# # Directory where your course files are stored
# course_directory = "temp files\jobs_txt"

# # Initialize tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
# model = AutoModel.from_pretrained("thenlper/gte-large")

# #This code was all for Batches.

# # Define batch size and number of batches
# batch_size = 25
# num_batches = (len(os.listdir(course_directory)) + batch_size - 1) // batch_size

# for batch_index in range(num_batches):
#     # Initialize list to store batch embeddings
#     batch_embeddings = []

#     # Iterate over files in the batch
#     for filename in os.listdir(course_directory)[batch_index * batch_size : (batch_index + 1) * batch_size]:
#         print(filename)
#         if filename.endswith(".txt"):
#             # Read the contents of the text file
#             file_path = os.path.join(course_directory, filename)
#             with open(file_path, "r", encoding="utf-8") as file:
#                 course_content = file.read()

#             # Tokenize content
#             tokenized_content = tokenizer(course_content, max_length=512, padding=True, truncation=True, return_tensors='pt')

#             # Pass tokenized content through the model
#             with torch.no_grad():  # No need to compute gradients during inference
#                 outputs = model(**tokenized_content)
#                 embeddings = outputs.last_hidden_state

#             # Average pooling to get a single embedding for each course description
#             attention_mask = tokenized_content['attention_mask']
#             pooled_embedding = embeddings.mean(dim=1)

#             # Normalize the embeddings
#             normalized_embedding = F.normalize(pooled_embedding, p=2, dim=1)

#             # Append the normalized embedding to the batch list
#             batch_embeddings.append(normalized_embedding)

#     # Concatenate embeddings in the batch
#     batch_embeddings_tensor = torch.cat(batch_embeddings, dim=0)

#     # Save the batch embeddings to a .pt file
#     torch.save(batch_embeddings_tensor, f'temp files/Embedding/_{batch_index * batch_size}.pt')



    
# print("Embeddings saved successfully.")



































# import torch
# import os

# # Directory where the .pt files are stored
# pt_files_directory = 'temp files/Embedding2/'

# # List all .pt files in the directory
# pt_files = [file for file in os.listdir(pt_files_directory) if file.endswith('.pt')]

# # Initialize an empty list to store embeddings
# embeddings_list = []

# # Iterate over each .pt file and load its contents
# for pt_file in pt_files:
#     file_path = os.path.join(pt_files_directory, pt_file)
#     embeddings = torch.load(file_path)
#     embeddings_list.append(embeddings)

# # Concatenate embeddings from all files
# all_embeddings = torch.cat(embeddings_list, dim=0)

# # Save the merged embeddings to a single .pt file
# merged_file_path = 'temp files/merged_embeddings.pt'
# torch.save(all_embeddings, merged_file_path)

# print("Merged embeddings saved successfully.")


import torch
import os

# Directory where the .pt files are stored
pt_files_directory = 'temp files\Embedding\en'

# List all .pt files in the directory
pt_files = [file for file in os.listdir(pt_files_directory) if file.endswith('.pt')]

# Initialize an empty list to store embeddings
embeddings_list = []

# Iterate over each .pt file and load its contents
for pt_file in pt_files:
    file_path = os.path.join(pt_files_directory, pt_file)
    embeddings = torch.load(file_path)
    embeddings_list.append(embeddings)

# Concatenate embeddings from all files
all_embeddings = torch.cat(embeddings_list, dim=0)

# Save the merged embeddings to a single .pt file
merged_file_path = 'temp files\Embedding\en\merged_embeddings_de.pt'
torch.save(all_embeddings, merged_file_path)

print("Merged embeddings saved successfully.")
