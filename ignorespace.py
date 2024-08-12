import re

def find_with_original_indices(text, search_query):
    # Normalize whitespace in both the text and the search query
    normalized_text = re.sub(r'\s+', ' ', text)
    normalized_query = re.sub(r'\s+', ' ', search_query)
    
    # Search for the query in the normalized text
    match = re.search(re.escape(normalized_query), normalized_text)
    
    if match:
        # Start and end indices in the normalized text
        norm_start = match.start()
        norm_end = match.end()

        # Map back to the original text
        original_start, original_end = None, None
        current_norm_index = 0
        current_original_index = 0

        for i, char in enumerate(text):
            if current_norm_index == norm_start and original_start is None:
                original_start = current_original_index

            if current_norm_index == norm_end:
                original_end = current_original_index
                break

            if not char.isspace():
                current_norm_index += 1
            
            current_original_index += 1

        if original_start is not None and original_end is None:
            original_end = len(text)
        
        return original_start, original_end
    else:
        return None

# Example usage
document = "This        is a   sample    document."
search_text = "is a sample"

result = find_with_original_indices(document, search_text)

if result:
    start, end = result
    matched_text = document[start:end]
    print(f"Match found from index {start} to {end}: '{matched_text}'")
else:
    print("No match found")
