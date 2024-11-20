import os
import requests
import time

# Define the base URL for the OpenAlex API
BASE_URL = 'https://api.openalex.org/works'

# Define the search parameters
params = {
    'search': 'Judicial Review in European Legal Systems',
    'filter': 'is_oa:true,publication_year:2010-2024,has_abstract:true',
    'per-page': 25,  # Number of results per page
    'mailto': 'lyaminky2@gmail.com'  # Replace with your email
}

# Define the directory to save abstracts
save_dir = 'data/raw/legal'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Initialize counters
saved_count = 0
page = 1

while saved_count < 100:
    # Update the page number in parameters
    params['page'] = page

    # Send a GET request to the OpenAlex API
    response = requests.get(BASE_URL, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        works = data.get('results', [])

        # If no more works are found, break the loop
        if not works:
            print('No more works found.')
            break

        for work in works:
            if saved_count >= 100:
                break

            # Extract the abstract
            abstract_inverted_index = work.get('abstract_inverted_index', None)
            if abstract_inverted_index:
                # Get the abstract directly if available
                abstract_text = work.get('abstract')
                if not abstract_text:  # Fallback to inverted index if needed
                    # Create a list of words based on their positions
                    word_positions = []
                    for word, positions in abstract_inverted_index.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    
                    # Sort by position and extract just the words
                    sorted_words = [word for _, word in sorted(word_positions)]
                    
                    # Remove consecutive duplicates
                    cleaned_words = []
                    for word in sorted_words:
                        if not cleaned_words or word != cleaned_words[-1]:
                            cleaned_words.append(word)
                    
                    abstract_text = ' '.join(cleaned_words)

                # Define the filename
                filename = f'openalex_{saved_count + 1}.txt'
                file_path = os.path.join(save_dir, filename)

                # Save the abstract to a text file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(abstract_text)

                saved_count += 1
                print(f'Saved {filename}')

        # Increment the page number for the next iteration
        page += 1

        # Respectful delay to avoid hitting rate limits
        time.sleep(1)
    else:
        print(f'Error: {response.status_code}')
        break

print(f'Total abstracts saved: {saved_count}') 