#!/usr/bin/env python3

from langchain.tools import tool

@tool("search_inside_file")
def search_database(query:str, limit:int)->str:
    """
    This tool uses query and limit
    query:str which is the key word to find in the file
    limit:int provides the limit for slicing.
    """
    with open("search.txt","r") as file:
        data = file.readlines()
    
    for search in data[:limit]:
        if query == search:
            return f"Found {limit} results for {query}"
    
    return f"Nothing matches {query} for {limit}"
