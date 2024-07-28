from transformers import pipeline

# Load the pre-trained model
generator = pipeline('text-generation', model='gpt-2')

# Generate text
prompt = "Once upon a time,"
generated_text = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print(generated_text[0]['generated_text'])
