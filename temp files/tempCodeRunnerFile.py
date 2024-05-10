def average_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# course_descriptions_list = []

# for key in course_descriptions:
#     course_descriptions_list.append(key)

# print(course_descriptions_list)

# # Initialize tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
# model = AutoModel.from_pretrained("thenlper/gte-large")

# # Tokenize course descriptions
# course_batch_dict = tokenizer(course_descriptions_list, max_length=512, padding=True, truncation=True, return_tensors='pt')

# # Pass course descriptions through the model
# course_outputs = model(**course_batch_dict)
# course_embeddings = average_pool(course_outputs.last_hidden_state, course_batch_dict['attention_mask'])
# course_embeddings = F.normalize(course_embeddings, p=2, dim=1)

# # Get user prompt
# user_prompt = input("Enter your prompt: ")

# # Tokenize the prompt
# prompt_tokenized = tokenizer(user_prompt, max_length=512, padding=True, truncation=True, return_tensors='pt')

# # Compute embeddings for the prompt
# prompt_outputs = model(**prompt_tokenized)
# prompt_embeddings = average_pool(prompt_outputs.last_hidden_state, prompt_tokenized['attention_mask'])
# prompt_embeddings = F.normalize(prompt_embeddings, p=2, dim=1)

# # Calculate similarity scores
# scores = torch.matmul(prompt_embeddings, course_embeddings.T)
# scores = scores.squeeze()  # Remove extra dimensions

# # Rank courses based on similarity scores
# top_k = 2  # Number of top courses to recommend
# top_indices = scores.argsort(descending=True)[:top_k]
# recommended_courses = [course_descriptions_list[i] for i in top_indices]

# # Present recommendations
# print("Recommended Courses:")
# for i, course in enumerate(recommended_courses, 1):
#     print(f"{i}. {course}")











