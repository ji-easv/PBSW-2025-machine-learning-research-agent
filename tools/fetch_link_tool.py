import requests


def fetch_link(url: str) -> str:
    """Fetches the content of the given URL.

    Args:
        url (str): The URL to fetch.
    """

    try:
        response = requests.get(url)
        response.raise_for_status()

        return response.text
    except requests.RequestException as e:
        return f"Error fetching the URL: {e}"
