import json
import re

def product_ideas_to_json(input_str: str) -> str:
    """
    Converts a string representation of ProductIdea objects into JSON.
    """
    # Extract the part after "product_ideas="
    if '=' in input_str:
        _, list_str = input_str.split('=', 1)
    else:
        list_str = input_str

    # Convert ProductIdea(...) into {...}
    list_str = list_str.replace('ProductIdea(', '{').replace(')', '}')

    # Convert single quotes to double quotes
    list_str = list_str.replace("'", '"')

    # Convert key=value to key: value using regex
    # This targets key=value patterns inside the braces
    list_str = re.sub(r'(\w+)=', r'"\1": ', list_str)

    try:
        # Safely parse the string into Python objects
        ideas_list = json.loads(list_str)
    except Exception as e:
        raise ValueError(f"Failed to parse input string: {e}")

    # Return JSON string
    return ideas_list