import re

def find_ignore_whitespace(text, search_query):
    # normalize whitespace in the text and the query
    normalized_text = re.sub(r'\s+', ' ', text)
    normalized_query = re.sub(r'\s+', ' ', search_query)
    
    # search for the query in the normalized
    match = re.search(re.escape(normalized_query), normalized_text)
    
    if match:
        # find start and end of the match in the original text
        start_index = match.start()
        end_index = match.end()
        return start_index, end_index
    else:
        return None

# ex
document = "This    is a   sample    document."
search_text = "is a sample"

result = find_ignore_whitespace(document, search_text)

if result:
    start, end = result
    print(f"Match found at: {start} to {end}")
else:
    print("No match found")