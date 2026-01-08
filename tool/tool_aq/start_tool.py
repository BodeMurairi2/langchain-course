#!/usr/bin/env python3

from langchain.tools import tool

@tool(
    "search_inside_file",
    return_direct=False
)
def search_database(query: str, limit: int = 1000) -> str:
    """
    Search for a keyword or phrase inside a text file.

    Use this tool when you need to:
    - Check whether a word or phrase exists in the database
    - Retrieve lines that mention a specific topic
    - Gather context before explaining or summarizing information

    Arguments:
    - query (str): The keyword or phrase to search for (e.g. "AI", "artificial intelligence").
      This search is case-insensitive and matches partial words.
    - limit (int, optional): The maximum number of lines to scan from the file.
      Use a large value (e.g. 500–2000) to search most or all of the file.

    Returns:
    - A string containing the matching lines from the file (up to 5 results),
      or a clear message if no matches are found.
    """
    results = []

    with open("search.txt", "r") as file:
        data = file.readlines()

    for line in data[:limit]:
        if query.lower() in line.lower():
            results.append(line.strip())

    if results:
        return "\n".join(results[:5])

    return f"No matches found for '{query}' in the first {limit} lines."
