#!/usr/bin/env python3

from langchain.tools import tool

@tool
def search_database(query_search:str, limit:int = 10)->str:
    """
    Docstring for search_database
    
    Args:
        :param query_search: Description
        :type query_search: str
        :param limit: Description
        :type limit: int
    :return: Description
    :rtype: str
    """
    return f"found in {limit} result for {query_search}"

@tool("web_search")
def search(query:str)->str:
    """search in the web"""
    return f"Result found in {query}"

@tool("calculator", description="Perform arithmetic calculations.")
def calculator(expression:str)->str:
    """
    Docstring for calculator
    
    :param expression: Description
    :type expression: str
    :return: Description
    :rtype: str
    """